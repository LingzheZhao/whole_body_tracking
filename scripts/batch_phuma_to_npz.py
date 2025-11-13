"""Batch convert PHUMA motions (.npy/.npz) to npz and upload/link to Weights & Biases.

Overview
- Discovers input files (.npy/.npz) in a directory (optionally recursive) or a single file.
- Replays motion in Isaac/IsaacLab to export per-sample npz preserving relative folder structure.
- Uploads exported npz artifacts to a WandB registry project and links them there.
- Supports Wandb entity vs organization handling by using a non-prefixed canonical registry name.
- Supports saving exported npz files under '--output_dir' mirroring input structure.
- Supports parallelism by increasing '--upload_workers', '--sync_workers' to alleviate network bottlenecks.
- Automatically sanitize input file names from special characters.

Resumable Processing
- processed_motions.txt records relative paths (with extensions). On restart, those entries are skipped.
- Entries are appended immediately after local save to reflect real-time progress.
- File writes use a best-effort POSIX lock to reduce risk of corruption under multi-process/multi-instance writes.
- Supports startup synchronization with WandB to hydrate processed list and download locally-missing npz.
- Supports verification of uploaded entries against WandB registry (writes a remotely-missing list).
- Final check of any missing entries against the input dataset and re-process them.


Key Arguments
- --input_dir: File or directory containing PHUMA .npy/.npz.
- --recursive/--no-recursive: Whether to walk subdirectories.
- --input_fps / --output_fps / --frame_range: I/O frame rates and inclusive slicing.
- --output_dir: Root directory to save local npz files mirroring input relative paths.
- --upload_workers: Parallel uploader pool size (network I/O concurrency).

WandB Settings
- --wandb_entity: Run owner entity (user or team). Note: Runs cannot be created under an organization.
- --wandb_registry_entity: Registry owner entity (organization/team), e.g., 'cvgl-org'.
- --wandb_registry_project: Registry project to link artifacts (e.g., 'wandb-registry-Motions').
- Behavior:
  - The script initializes the run under a non-org entity (tries --wandb_entity, its '-org' stripped form, then default).
  - The artifact is then linked to the registry path:
      '<wandb_registry_entity>/<wandb_registry_project>/<encoded_rel_name>'
    If --wandb_registry_entity is not provided, falls back to '<wandb_registry_project>/<encoded_rel_name>'.
  - Encoded name preserves folders by replacing '/' with '__'.

WandB Synchronization (disabled by default)
- --sync_from_wandb:
  - On startup, sync from WandB REGISTRY (artifact collections), not from runs:
    - Append any missing rel_path (decoded from collection names) into processed_motions.txt
    - Download missing npz into '--output_dir/<rel_path>.npz'
  - The downloader tries multiple fully-qualified artifact specs, including org/team variants:
    '{registry_entity}/{registry_project}/{name}:latest',
    '{wandb_entity}-org/{registry_project}/{name}:latest',
    '{wandb_entity}/{registry_project}/{name}:latest',
    and case variants, plus fallback to the run project if needed.

WandB Verification (disabled by default)
- --verify_uploaded:
  - If enabled, the script checks all entries in processed_motions.txt against the WandB registry.
  - Missing entries are written to '--missing_list_name' in the input root.
- --missing_list_name:
  - Name of the report file for missing entries. Default: 'missing_in_wandb.txt'.

Resumable Processing
- processed_motions.txt records relative paths (with extensions). On restart, those entries are skipped.
- Entries are appended immediately after local save to reflect real-time progress.
- File writes use a best-effort POSIX lock to reduce risk of corruption under multi-process/multi-instance writes.

Notices
- Entity vs Organization:
  - WandB prohibits creating runs directly under organizations. Use a user/team entity for runs
    (e.g., 'cvgl'), and a separate org/team entity for the registry (e.g., 'cvgl-org').
  - If your visible registry full name is 'cvgl-org/wandb-registry-Motions/...', pass:
      --wandb_entity cvgl --wandb_registry_entity cvgl-org --wandb_registry_project wandb-registry-Motions
- Disk usage:
  - Exported npz files are saved under '--output_dir' mirroring input structure. Ensure enough free space.
- Performance:
  - Increase '--upload_workers' to alleviate network bottlenecks. Simulation remains single-process by default.
- Safety:
  - If multiple script instances point to the same processed_motions.txt, locking reduces but cannot fully
    eliminate cross-process race conditions on non-POSIX systems.
"""
from __future__ import annotations


PROJECT_NAME = "csv_to_npz"
REGISTRY_NAME = "Motions"

from wb_batch.sanitizer import sanitize_basename as _sanitize_basename
from wb_batch.filelock import append_with_lock as _append_with_lock
from wb_batch.wandb_client import WandbRegistry as WBWandbRegistry
from wb_batch.processed_list import ProcessedList as ProcessedListManager


# Old inline WandbRegistry removed in favor of wb_batch.wandb_client.WandbRegistry (WBWandbRegistry)


class MotionExporter:
    """Export one motion to npz and record processed entry immediately after local save."""

    def __init__(
        self,
        sim: sim_utils.SimulationContext,
        scene: InteractiveScene,
        joint_names: list[str],
        output_dir: Path,
        processed: ProcessedListManager,
    ):
        self.sim = sim
        self.scene = scene
        self.joint_names = joint_names
        self.output_dir = output_dir
        self.processed = processed

    def export(self, motion_file: Path, rel_entry: str) -> Path | None:
        return _run_single_motion(
            sim=self.sim,
            scene=self.scene,
            joint_names=self.joint_names,
            motion_file=motion_file,
            rel_entry=rel_entry,
            output_dir=self.output_dir,
            out_txt=self.processed.path,
        )


class BatchPhumaProcessor:
    """High-level batch processor for PHUMA conversion with parallel uploads."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.input_dir = Path(args.input_dir)
        self.base_root = self.input_dir if self.input_dir.is_dir() else self.input_dir.parent
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed = ProcessedListManager(self.input_dir)
        self.wandb = WBWandbRegistry(args.wandb_entity, args.wandb_registry_project, args.wandb_registry_entity)
        # Read-only backup of all input motions' relative paths, populated in discover_inputs()
        self._all_inputs_rel: list[str] = []

    def _rename_if_needed(self, p: Path) -> Path:
        """Rename the given file if its basename contains disallowed characters or '__' sequence."""
        if not p.exists() or not p.is_file():
            return p
        sanitized_name = _sanitize_basename(p.name)
        # avoid '__' by collapsing underscores already done in _sanitize_basename
        if sanitized_name == p.name:
            return p
        target = p.with_name(sanitized_name)
        # Avoid collision by adding numeric suffix if needed
        if target.exists():
            stem, suffix = target.stem, target.suffix
            k = 1
            while target.exists():
                candidate = p.with_name(f"{stem}-{k}{suffix}")
                target = candidate
                k += 1
        try:
            p.rename(target)
            print(f"[INFO]: Renamed input file: {p.name} -> {target.name}")
            return target
        except Exception as e:
            print(f"[WARN]: Failed to rename {p.name}: {e}. Proceeding with original name.")
            return p

    def _compute_missing_and_update(self) -> set[str]:
        missing_set: set[str] = set()
        if not self.args.verify_uploaded:
            return missing_set
        cfg = self.wandb.serialize()
        print(f"[INFO]: Verifying uploaded entries against WandB registry (cfg={cfg})")
        available_rels = WBWandbRegistry.list_all_rel_entries_with_cfg(cfg)
        rels_sorted = sorted(self.processed.processed_rel_set)
        missing: list[str] = []
        for rel in _progress_iter(rels_sorted, total=len(rels_sorted), desc="Compare with registry"):
            if rel not in available_rels:
                missing.append(rel)
        missing_path = (self.processed.path.parent / self.args.missing_list_name)
        try:
            with open(missing_path, "w", encoding="utf-8") as f:
                for rel in missing:
                    f.write(rel + "\n")
            print(f"[INFO]: Verified processed entries. Missing in WandB: {len(missing)}. Saved to {missing_path}")
        except Exception as e:
            print(f"[WARN]: Failed to write missing list: {e}")
        if missing:
            self.processed.backup_and_remove(set(missing))
            missing_set = set(missing)
        return missing_set

    def _prioritize_files(self, files: list[Path], missing_set: set[str]) -> list[Path]:
        if not missing_set:
            return files

        def _priority_key(p: Path) -> tuple[int, str]:
            rel = p.relative_to(self.base_root).as_posix()
            return (0 if rel in missing_set else 1, rel)

        print(f"[INFO]: Prioritizing {len(files)}")
        return sorted(files, key=_priority_key)

    def _process_and_upload(self, exporter: "MotionExporter", files: list[Path], desc: str = "Uploading to WandB") -> int:
        import multiprocessing as _mp  # avoid early import issues
        pool = _mp.Pool(processes=max(1, int(self.args.upload_workers)))
        pending: list[tuple[str, "_mp.pool.ApplyResult"]] = []  # type: ignore[name-defined]
        for f in files:
            rel = f.relative_to(self.base_root).as_posix()
            if self.processed.contains(rel):
                continue
            saved = exporter.export(f, rel)
            if saved is not None:
                pending.append((rel, self.wandb.upload_async(pool, saved, rel)))
        pool.close()
        uploaded = 0
        for _rel, _res in pending:
            break
        from wb_batch.pool import drain_results as _drain_results
        for rel, ok in _drain_results(pending, total=len(pending), desc=desc):
            if not ok:
                print(f"[WARN]: Upload failed for {rel}")
            if ok:
                uploaded += 1
            else:
                # Roll back processed list entry if upload failed
                try:
                    self.processed.remove(rel)
                except Exception as _e:
                    print(f"[WARN]: Failed to roll back processed entry {rel}: {_e}")
        pool.join()
        return uploaded

    def _final_verify_and_reprocess(self, exporter: "MotionExporter") -> None:
        if not self.args.verify_uploaded:
            return
        cfg = self.wandb.serialize()
        available_rels = WBWandbRegistry.list_all_rel_entries_with_cfg(cfg)
        # Use the full list of motions discovered from the input dataset
        rels_full = list(self._all_inputs_rel) if isinstance(self._all_inputs_rel, list) else []
        if not rels_full:
            # Fallback to processed set if full list is unavailable
            rels_full = list(self.processed.processed_rel_set)
        rels_sorted = sorted(rels_full)
        final_missing: list[str] = []
        for rel in _progress_iter(rels_sorted, total=len(rels_sorted), desc="Final compare with registry"):
            if rel not in available_rels:
                final_missing.append(rel)
        if not final_missing:
            return
        print(f"[INFO]: Final pass: {len(final_missing)} motion(s) still missing on WandB. Re-processing now...")
        files: list[Path] = []
        for rel in final_missing:
            src = (self.base_root / rel)
            if src.exists():
                files.append(src)
        if not files:
            print("[WARN]: Final pass: no source files present for missing motions.")
            return
        uploaded2 = self._process_and_upload(exporter, files, desc="Uploading (final pass)")
        print(f"[INFO]: Final pass uploaded {uploaded2} item(s).")

    def discover_inputs(self) -> list[Path]:
        files: list[Path] = []
        if self.input_dir.is_file():
            assert self.input_dir.suffix.lower() in {".npy", ".npz"}
            files.append(self._rename_if_needed(self.input_dir))
        else:
            if self.args.recursive:
                for root, _, fnames in os.walk(self.input_dir):
                    root_path = Path(root)
                    for fn in fnames:
                        if fn.endswith(".npy") or fn.endswith(".npz"):
                            files.append(self._rename_if_needed(root_path / fn))
            else:
                for fn in os.listdir(self.input_dir):
                    if fn.endswith(".npy") or fn.endswith(".npz"):
                        files.append(self._rename_if_needed(self.input_dir / fn))
        files = sorted(files)
        # Read-only backup of full relative list for final verification usage
        try:
            self._all_inputs_rel = [p.relative_to(self.base_root).as_posix() for p in files]
        except Exception:
            self._all_inputs_rel = []
        assert len(files) > 0, f"No .npy/.npz files found in {self.input_dir}"
        print(f"[INFO]: Found {len(files)} PHUMA file(s) in {self.input_dir} (recursive={self.args.recursive})")
        return files

    def run(self, sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]) -> None:
        self.processed.load()
        if self.args.sync_from_wandb:
            cfg_sync = self.wandb.serialize()
            cfg_sync["sync_workers"] = args_cli.sync_workers
            _sync_processed_from_wandb(self.processed.path, self.processed.processed_rel_set, self.processed.processed_stem_set, self.output_dir, cfg_sync)
        # Compute missing and adjust processed list
        missing_set = self._compute_missing_and_update()
        # Prepare exporter and inputs
        exporter = MotionExporter(sim, scene, joint_names, self.output_dir, self.processed)
        files = self.discover_inputs()
        files = self._prioritize_files(files, missing_set)
        # Process and upload
        uploaded = self._process_and_upload(exporter, files, desc="Uploading to WandB")
        print(f"[INFO]: Saved processed list to {self.processed.path}")
        print(f"[INFO]: Successfully uploaded {uploaded} item(s).")
        # Remove missing list after prioritized processing
        missing_list_path = (self.processed.path.parent / self.args.missing_list_name)
        if missing_list_path.exists():
            try:
                missing_list_path.unlink()
                print(f"[INFO]: Removed missing list file {missing_list_path}")
            except Exception as e:
                print(f"[WARN]: Failed to remove missing list file: {e}")
        # Final validation and re-process remaining missing motions
        self._final_verify_and_reprocess(exporter)


import os
import argparse
import numpy as np
from pathlib import Path
from isaaclab.app import AppLauncher
import os as _os
from wb_batch.pool import progress_iter as _progress_iter

# Suppress W&B console noise
_os.environ.setdefault("WANDB_SILENT", "true")  # set to true to suppress wandb console output
# Suppress W&B console noise (valid values for WANDB_CONSOLE: auto/off/wrap/redirect/wrap_raw/wrap_emu)
_os.environ.setdefault("WANDB_CONSOLE", "off")


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
parser.add_argument("--upload_workers", type=int, default=4, help="Number of parallel upload workers.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="/tmp/motions_phuma",
    help="Directory to save intermediate npz files before uploading.",
)
parser.add_argument(
    "--sync_from_wandb",
    action="store_true",
    help="At startup, sync processed_motions.txt from WandB REGISTRY (artifact collections),"
    " and download missing npz files to --output_dir.",
)
parser.add_argument(
    "--wandb_registry_project",
    type=str,
    default="wandb-registry-Motions",
    help="W&B registry project that stores finalized artifacts (e.g., 'wandb-registry-Motions').",
)
parser.add_argument(
    "--wandb_registry_entity",
    type=str,
    default=None,
    help="W&B entity/org that owns the registry project (e.g., 'cvgl-org'). If not provided, we will try "
    "both --wandb_entity and a derived '<wandb_entity>-org' alias.",
)
parser.add_argument(
    "--wandb_sync_project",
    type=str,
    default=None,
    help="Deprecated: previously used to read runs; now unused since registry sync is default.",
)
parser.add_argument(
    "--wandb_sync_entity",
    type=str,
    default=None,
    help="Deprecated: previously used to read runs; now unused since registry sync is default.",
)
parser.add_argument(
    "--verify_uploaded",
    action="store_true",
    default=False,
    help="Verify entries in processed_motions.txt exist in WandB registry (writes a remotely-missing list).",
)
parser.add_argument(
    "--missing_list_name",
    type=str,
    default="missing_in_wandb.txt",
    help="Filename to write motions remotely-missing in WandB registry (relative to input root). Default: missing_in_wandb.txt",
)
parser.add_argument(
    "--sync_workers",
    type=int,
    default=8,
    help="Number of parallel workers to download locally-missing files during sync.",
)


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
    output_dir: Path,
    out_txt: Path,
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

            # Save to per-sample npz under output_dir preserving subfolder structure
            target_npz = output_dir / rel_entry
            target_npz = target_npz.with_suffix(".npz")
            target_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez(target_npz.as_posix(), **log)
            # Mark as processed immediately after local save
            _append_with_lock(out_txt, rel_entry + "\n")
            file_saved = True
            return target_npz


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    joint_names: list[str]
):
    """Run the batch processor to export and upload motions."""
    processor = BatchPhumaProcessor(args_cli)
    processor.run(sim, scene, joint_names)
    print("[INFO]: All motions processed.")
    simulation_app.close()


# shims removed in favor of direct imports from wb_batch.*


def _sync_processed_from_wandb(
    out_txt: Path, processed_rel_set: set[str], processed_stem_set: set[str], output_dir: Path, registry_cfg: dict | None = None
) -> None:
    """Sync processed list and local files from WandB REGISTRY (artifact collections).
    If an entry exists on WandB but the local output file is locally-missing, download it to output_dir/rel_path."""
    try:
        cfg = registry_cfg or {
            "entity": args_cli.wandb_entity,
            "registry_entity": args_cli.wandb_registry_entity,
            "registry_project": args_cli.wandb_registry_project,
            "run_project": PROJECT_NAME,
            "sync_workers": args_cli.sync_workers,
        }
        return WBWandbRegistry.sync_runs_with_cfg(out_txt, processed_rel_set, processed_stem_set, output_dir, cfg)
    except Exception as e:
        print(f"[WARN]: Failed to sync from wandb: {e}")


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
