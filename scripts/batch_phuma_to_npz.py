"""Batch convert PHUMA motions (.npy/.npz) to npz and upload to Weights & Biases.

Features
- Input can be a single .npy/.npz file or a directory. Supports --recursive to walk all subdirectories.
- For each motion, exports a /tmp/motion.npz artifact and uploads to wandb.
- Preserves subfolder information in wandb by encoding the relative path into the run/artifact name:
  'sub/dir/file.npy' -> 'sub__dir__file.npy'. The full relative path is also stored in run.config:
  - config.rel_path  (e.g., 'sub/dir/file.npy')
  - config.rel_dir   (e.g., 'sub/dir')
  - config.filename  (e.g., 'file.npy')
- Writes and appends to 'processed_motions.txt' in the input root. Each line is the relative path (with extension).
  On restart, previously processed entries are skipped to support resumable processing.
- Uses fps from PHUMA if present; otherwise falls back to --input_fps.
- Supports per-sample frame slicing via --frame_range START END (inclusive).
- Supports setting wandb entity via --wandb_entity.

Usage
.. code-block:: bash

    # Recursively process a directory tree of PHUMA files, 50 fps output
    python batch_phuma_to_npz.py \\
        --input_dir /path/to/phuma_root \\
        --recursive \\
        --output_fps 50 \\
        --wandb_entity <your_wandb_entity>

    # Process a single PHUMA file with frame slicing and custom fallback input fps
    python batch_phuma_to_npz.py \\
        --input_dir /path/to/seq.npy \\
        --input_fps 30 \\
        --frame_range 1 600 \\
        --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import os
import argparse
import numpy as np
from pathlib import Path
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from PHUMA (.npy/.npz) files and output to npz file.")
parser.add_argument("--input_dir", type=str, required=True, help="Path to PHUMA file (.npy/.npz) or directory.")
parser.add_argument("--input_fps", type=int, default=30, help="Fallback fps if file has no fps.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument("--recursive", action="store_true", help="Recursively walk subdirectories for PHUMA files.")
parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Do not recurse into subdirectories.")
parser.set_defaults(recursive=False)
parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity (user or team).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from PHUMA file."""
        path = Path(self.motion_file)
        assert path.suffix.lower() in {".npy", ".npz"}, "Input file must be .npy or .npz."
        motion_np, phuma_fps = self._load_phuma(path)
        # Override input fps if provided
        if phuma_fps is not None:
            self.input_fps = int(phuma_fps)
            self.input_dt = 1.0 / self.input_fps
        # Apply frame slicing if requested
        if self.frame_range is not None:
            start = max(self.frame_range[0] - 1, 0)
            end = self.frame_range[1]
            motion_np = motion_np[start:end, :]

        motion = torch.from_numpy(motion_np).to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # convert to wxyz
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _load_phuma(self, path: Path) -> tuple[np.ndarray, float | None]:
        if path.suffix.lower() == ".npz":
            d = dict(**np.load(path, allow_pickle=True))
        else:
            arr = np.load(path, allow_pickle=True)
            try:
                d = dict(enumerate(arr.flatten()))[0]
            except Exception:
                try:
                    d = arr.item()  # type: ignore[assignment]
                except Exception:
                    d = dict(arr)  # type: ignore[arg-type]

        root_trans = np.asarray(d["root_trans"]).astype(np.float32)
        root_ori = np.asarray(d["root_ori"]).astype(np.float32)  # (T,4) xyzw
        dof_pos = np.asarray(d["dof_pos"]).astype(np.float32)
        fps_val = d.get("fps", None)
        fps: float | None = float(fps_val) if fps_val is not None else None
        assert root_trans.shape[0] == root_ori.shape[0] == dof_pos.shape[0]
        motion_np = np.concatenate([root_trans, root_ori, dof_pos], axis=1).astype(np.float32)
        return motion_np, fps

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        bool,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def _run_single_motion(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    joint_names: list[str],
    motion_file: Path,
    rel_entry: str,
):
    """Runs the simulation loop for a single motion."""

    sim.reset()

    # Load motion
    motion = MotionLoader(
        motion_file=motion_file.as_posix(),
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while simulation_app.is_running() and not file_saved:
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            np.savez("/tmp/motion.npz", **log)

            import wandb

            # Preserve subfolder information via sanitized relative path
            COLLECTION = rel_entry.replace("/", "__").replace("\\", "__")
            try:
                run = wandb.init(
                    project="csv_to_npz",
                    name=COLLECTION,
                    entity=args_cli.wandb_entity,
                    reinit=True,
                    config={
                        "rel_path": rel_entry,
                        "rel_dir": str(Path(rel_entry).parent.as_posix()),
                        "filename": Path(rel_entry).name,
                    },
                )
                print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
                REGISTRY = "motions"
                logged_artifact = run.log_artifact(
                    artifact_or_path="/tmp/motion.npz", name=COLLECTION, type=REGISTRY
                )
                run.link_artifact(
                    artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}"
                )
                print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION} (rel={rel_entry})")
            except Exception as e:
                print(f"[WARN]: Failed to log to wandb for {COLLECTION}: {e}")
            finally:
                try:
                    run.finish()  # type: ignore[union-attr]
                except Exception:
                    pass
            file_saved = True


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    joint_names: list[str]
):
    """Runs the simulation loop."""
    input_files = []
    input_dir = Path(args_cli.input_dir)
    if input_dir.is_file():
        assert input_dir.suffix.lower() in {".npy", ".npz"}, "Input file must be .npy or .npz."
        input_files.append(input_dir)
    elif input_dir.is_dir():
        if args_cli.recursive:
            for root, _, files in os.walk(input_dir):
                root_path = Path(root)
                for file in files:
                    if file.endswith(".npy") or file.endswith(".npz"):
                        input_files.append(root_path / file)
        else:
            for file in os.listdir(input_dir):
                if file.endswith(".npy") or file.endswith(".npz"):
                    input_files.append(input_dir / file)
    else:
        raise ValueError(f"Input path {input_dir} is not a file or directory.")

    input_files = sorted(input_files)  # TODO: change to natural sort if needed
    assert len(input_files) > 0, f"No .npy/.npz files found in {input_dir}"
    print(f"[INFO]: Found {len(input_files)} PHUMA file(s) in {input_dir} (recursive={args_cli.recursive})")

    # Load existing processed list for resume
    list_path = input_dir if input_dir.is_dir() else input_dir.parent
    out_txt = list_path / "processed_motions.txt"
    base_root = input_dir if input_dir.is_dir() else input_dir.parent
    processed_rel_set = set()
    processed_stem_set = set()
    if out_txt.exists():
        try:
            with open(out_txt, "r", encoding="utf-8") as f:
                for line in f:
                    entry = line.strip()
                    if entry:
                        processed_rel_set.add(entry)
                        processed_stem_set.add(Path(entry).stem)
            print(f"[INFO]: Loaded {len(processed_rel_set)} processed entries from {out_txt}")
        except Exception as e:
            print(f"[WARN]: Failed to read processed list {out_txt}: {e}")

    for input_file in input_files:
        rel_entry = input_file.relative_to(base_root).as_posix()
        if rel_entry in processed_rel_set or input_file.stem in processed_stem_set:
            continue
        _run_single_motion(sim, scene, joint_names, input_file, rel_entry)
        # Append to processed list incrementally
        try:
            with open(out_txt, "a", encoding="utf-8") as f:
                f.write(rel_entry + "\n")
        except Exception as e:
            print(f"[WARN]: Failed to append processed name: {e}")

    print(f"[INFO]: Saved processed list to {out_txt}")

    print("[INFO]: All motions processed.")
    # close sim app
    simulation_app.close()


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(
        sim,
        scene,
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
