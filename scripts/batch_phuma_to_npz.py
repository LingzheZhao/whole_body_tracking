"""Batch convert PHUMA motions (.npy/.npz) to npz and upload/link to Weights & Biases.

Overview
- Discovers input files (.npy/.npz) in a directory (optionally recursive) or a single file.
- Replays motion in Isaac/IsaacLab to export per-sample npz preserving relative folder structure.
- Uploads exported npz artifacts to a WandB registry project and links them there.
- Maintains resumable progress via processed_motions.txt (relative paths, one per line).
- Supports startup synchronization with WandB to hydrate processed list and download missing npz.

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
- --sync_from_wandb:
  - On startup, sync from WandB REGISTRY (artifact collections), not from runs:
    - Append any missing rel_path (decoded from collection names) into processed_motions.txt
    - Download missing npz into '--output_dir/<rel_path>.npz'
  - The downloader tries multiple fully-qualified artifact specs, including org/team variants:
    '{registry_entity}/{registry_project}/{name}:latest',
    '{wandb_entity}-org/{registry_project}/{name}:latest',
    '{wandb_entity}/{registry_project}/{name}:latest',
    and case variants, plus fallback to the run project if needed.

Verification (enabled by default)
- --verify_uploaded / --no-verify-uploaded:
  - If enabled (default), the script checks all entries in processed_motions.txt against the WandB registry.
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
REGISTRY_NAME = "motions"


class ProcessedListManager:
    """Manage the processed_motions.txt with locking and fast membership checks."""

    def __init__(self, base_root: Path):
        self.base_root = base_root
        self.path = (base_root if base_root.is_dir() else base_root.parent) / "processed_motions.txt"
        self.processed_rel_set: set[str] = set()
        self.processed_stem_set: set[str] = set()

    def load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    for line in f:
                        entry = line.strip()
                        if entry:
                            self.processed_rel_set.add(entry)
                            self.processed_stem_set.add(Path(entry).stem)
                print(f"[INFO]: Loaded {len(self.processed_rel_set)} processed entries from {self.path}")
            except Exception as e:
                print(f"[WARN]: Failed to read processed list {self.path}: {e}")

    def append(self, rel_entry: str) -> None:
        if rel_entry not in self.processed_rel_set:
            _append_with_lock(self.path, rel_entry + "\n")
            self.processed_rel_set.add(rel_entry)
            self.processed_stem_set.add(Path(rel_entry).stem)

    def contains(self, rel_entry: str) -> bool:
        return rel_entry in self.processed_rel_set or Path(rel_entry).stem in self.processed_stem_set


class WandbRegistry:
    """Encapsulate Weights & Biases interactions (upload and sync)."""

    def __init__(self, entity: str | None, registry_project: str, registry_entity: str | None):
        self.entity = entity
        self.registry_project = registry_project
        self.registry_entity = registry_entity
        # Non-prefixed canonical registry name (UI: orgs/<org>/registry/<Name>)
        if isinstance(registry_project, str) and registry_project.startswith("wandb-registry-"):
            self.registry_name = registry_project[len("wandb-registry-") :]
        else:
            self.registry_name = registry_project
        # Common project name used to create runs
        self.run_project = PROJECT_NAME

    # ---------------------------- helpers (paths/entities) ----------------------------
    def build_collection_name(self, rel_entry: str) -> str:
        return rel_entry.replace("/", "__").replace("\\", "__")

    def build_target_path(self, collection: str) -> str:
        """Prefer linking to '<entity>/registry/<Name>/<collection>' using non-prefixed registry name."""
        if self.registry_entity:
            return f"{self.registry_entity}/registry/{self.registry_name}/{collection}"
        return f"registry/{self.registry_name}/{collection}"

    def candidate_run_entities(self) -> list[str | None]:
        candidates: list[str | None] = []
        if self.entity:
            candidates.append(self.entity)
            if self.entity.endswith("-org"):
                candidates.append(self.entity[:-4])
        candidates.append(None)  # default logged-in
        # deduplicate while preserving order
        seen = set()
        return [e for e in candidates if not (e in seen or seen.add(e))]

    def candidate_artifact_specs(self, collection: str, fallback_project: str) -> list[str]:
        entities: list[str] = []
        if self.registry_entity:
            entities.append(self.registry_entity)
        if self.entity:
            entities.append(self.entity)
            entities.append(f"{self.entity}-org")
        seen = set()
        entities = [e for e in entities if e and not (e in seen or seen.add(e))]
        specs: list[str] = []
        for ent in entities:
            # Preferred: registry paths
            specs.append(f"{ent}/registry/{self.registry_name}/{collection}:latest")
            # Back-compat: direct project names
            specs.append(f"{ent}/{self.registry_project}/{collection}:latest")
            specs.append(f"{ent}/wandb-registry-Motions/{collection}:latest")
            specs.append(f"{ent}/wandb-registry-motions/{collection}:latest")
            specs.append(f"{ent}/{fallback_project}/{collection}:latest")
        specs.append(f"registry/{self.registry_name}/{collection}:latest")
        specs.append(f"{self.registry_project}/{collection}:latest")
        specs.append(f"wandb-registry-Motions/{collection}:latest")
        specs.append(f"wandb-registry-motions/{collection}:latest")
        specs.append(f"{fallback_project}/{collection}:latest")
        return specs

    def serialize(self) -> dict:
        return {
            "entity": self.entity,
            "registry_project": self.registry_project,
            "registry_entity": self.registry_entity,
            "registry_name": self.registry_name,
            "run_project": self.run_project,
        }

    # ---------------------------- public API (used by caller) ----------------------------
    def upload_async(self, pool: "mp.pool.Pool", npz_path: Path, rel_entry: str) -> "mp.pool.ApplyResult":  # type: ignore[name-defined]
        cfg = self.serialize()
        return pool.apply_async(_uploader_task, args=(npz_path.as_posix(), rel_entry, cfg))

    def sync(self, processed: "ProcessedListManager", output_dir: Path) -> None:
        cfg = self.serialize()
        _sync_processed_from_wandb(processed.path, processed.processed_rel_set, processed.processed_stem_set, output_dir, cfg)

    # ---------------------------- static utilities (used in workers) ----------------------------
    @staticmethod
    def upload_with_cfg(npz_path: str, rel_entry: str, cfg: dict) -> bool:
        import wandb
        collection = rel_entry.replace("/", "__").replace("\\", "__")
        # choose run entity
        entity = cfg.get("entity")
        registry_entity = cfg.get("registry_entity")
        registry_name = cfg.get("registry_name", cfg.get("registry_project"))
        run_project = cfg.get("run_project", PROJECT_NAME)
        candidate_run_entities: list[str | None] = []
        if entity:
            candidate_run_entities.append(entity)
            if isinstance(entity, str) and entity.endswith("-org"):
                candidate_run_entities.append(entity[:-4])
        candidate_run_entities.append(None)
        run = None
        last_err: Exception | None = None
        for run_entity in [e for i, e in enumerate(candidate_run_entities) if e not in candidate_run_entities[:i]]:
            try:
                run = wandb.init(
                    project=run_project,
                    name=collection,
                    entity=run_entity,
                    reinit=True,
                    config={
                        "rel_path": rel_entry,
                        "rel_dir": str(Path(rel_entry).parent.as_posix()),
                        "filename": Path(rel_entry).name,
                    },
                )
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        if run is None:
            print(f"[WARN][upload] Failed to init wandb run for {collection}: {last_err}")
            return False
        print(f"[INFO][upload] Logging motion to wandb: {collection}")
        logged_artifact = run.log_artifact(artifact_or_path=npz_path, name=collection, type="motions")
        # target path: prefer registry path with non-prefixed name
        if registry_entity:
            target_path = f"{registry_entity}/registry/{registry_name}/{collection}"
        else:
            target_path = f"registry/{registry_name}/{collection}"
        run.link_artifact(artifact=logged_artifact, target_path=target_path)
        print(f"[INFO][upload] Motion saved to wandb registry: {target_path} (rel={rel_entry})")
        try:
            run.finish()
        except Exception:
            pass
        return True

    @staticmethod
    def download_with_cfg(rel_entry: str, target_npz: Path, cfg: dict) -> bool:
        import wandb
        api = wandb.Api()
        collection = rel_entry.replace("/", "__").replace("\\", "__")
        registry_project = cfg.get("registry_project")
        registry_name = cfg.get("registry_name", registry_project)
        entity = cfg.get("entity")
        registry_entity = cfg.get("registry_entity")
        run_project = cfg.get("run_project", PROJECT_NAME)
        # build candidates
        probe_entities: list[str] = []
        if registry_entity:
            probe_entities.append(registry_entity)
        if entity:
            probe_entities.append(entity)
            probe_entities.append(f"{entity}-org")
        seen = set()
        probe_entities = [e for e in probe_entities if e and not (e in seen or seen.add(e))]
        candidates: list[str] = []
        for ent in probe_entities:
            candidates.append(f"{ent}/registry/{registry_name}/{collection}:latest")
            candidates.append(f"{ent}/{registry_project}/{collection}:latest")
            candidates.append(f"{ent}/wandb-registry-Motions/{collection}:latest")
            candidates.append(f"{ent}/wandb-registry-motions/{collection}:latest")
            candidates.append(f"{ent}/{run_project}/{collection}:latest")
        candidates.append(f"registry/{registry_name}/{collection}:latest")
        candidates.append(f"{registry_project}/{collection}:latest")
        candidates.append(f"wandb-registry-Motions/{collection}:latest")
        candidates.append(f"wandb-registry-motions/{collection}:latest")
        candidates.append(f"{run_project}/{collection}:latest")
        artifact = None
        for spec in candidates:
            try:
                artifact = api.artifact(spec)
                break
            except Exception:
                artifact = None
        if artifact is None:
            print(f"[WARN]: Artifact not found on wandb for {collection}. Tried: {candidates}")
            return False
        with tempfile.TemporaryDirectory() as tmpdir:
            dl_dir = Path(artifact.download(root=tmpdir))
            found = None
            for p in dl_dir.rglob("*.npz"):
                found = p
                break
            if found is None:
                print(f"[WARN]: No .npz found in artifact {collection}")
                return False
            target_npz.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(found.as_posix(), target_npz.as_posix())
        return True

    @staticmethod
    def artifact_exists_with_cfg(rel_entry: str, cfg: dict) -> bool:
        """Return True if the artifact for rel_entry exists in WandB registry."""
        import wandb
        api = wandb.Api()
        collection = rel_entry.replace("/", "__").replace("\\", "__")
        registry_project = cfg.get("registry_project")
        entity = cfg.get("entity")
        registry_entity = cfg.get("registry_entity")
        run_project = cfg.get("run_project", PROJECT_NAME)
        probe_entities: list[str] = []
        if registry_entity:
            probe_entities.append(registry_entity)
        if entity:
            probe_entities.append(entity)
            probe_entities.append(f"{entity}-org")
        seen = set()
        probe_entities = [e for e in probe_entities if e and not (e in seen or seen.add(e))]
        candidates: list[str] = []
        for ent in probe_entities:
            candidates.append(f"{ent}/{registry_project}/{collection}:latest")
            candidates.append(f"{ent}/wandb-registry-Motions/{collection}:latest")
            candidates.append(f"{ent}/wandb-registry-motions/{collection}:latest")
            candidates.append(f"{ent}/{run_project}/{collection}:latest")
        candidates.append(f"{registry_project}/{collection}:latest")
        candidates.append(f"wandb-registry-Motions/{collection}:latest")
        candidates.append(f"wandb-registry-motions/{collection}:latest")
        candidates.append(f"{run_project}/{collection}:latest")
        for spec in candidates:
            try:
                _ = api.artifact(spec)
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def sync_runs_with_cfg(
        out_txt: Path, processed_rel_set: set[str], processed_stem_set: set[str], output_dir: Path, cfg: dict
    ) -> None:
        import wandb
        api = wandb.Api()
        print("[INFO]: Syncing from WandB registry (collections) ...")
        print(f"[INFO]: cfg: {cfg}")
        print(f"[INFO]: Trying to list collections via registries API, this may take a few minutes...")
        # Try registries API first
        registry_project = cfg.get("registry_project")
        registry_name = cfg.get("registry_name", registry_project)
        entity = cfg.get("entity")
        registry_entity = cfg.get("registry_entity")
        # 1) registries API
        collections = WandbRegistry._try_list_collections_via_registries(api, registry_name, registry_entity)
        if collections is not None:
            WandbRegistry._process_collections_parallel(collections, out_txt, processed_rel_set, processed_stem_set, output_dir, cfg)
            return
        # 2) artifact_type probing fallback
        collections = WandbRegistry._try_list_collections_via_artifact_type(api, entity, registry_entity, registry_project, registry_name)
        if collections is not None:
            WandbRegistry._process_collections_parallel(collections, out_txt, processed_rel_set, processed_stem_set, output_dir, cfg)
            return
        # 3) runs fallback
        WandbRegistry._fallback_sync_via_runs(api, out_txt, processed_rel_set, processed_stem_set, output_dir, cfg)

    @staticmethod
    def _try_list_collections_via_registries(api, registry_name: str | None, registry_entity: str | None):
        try:
            regs = api.registries(filter={"name": registry_name}) if registry_name else api.registries()
            reg_obj = None
            for r in regs:
                try:
                    ent = getattr(r, "entity", None) or getattr(r, "organization_name", None)
                except Exception:
                    ent = None
                if registry_entity is None or ent == registry_entity:
                    reg_obj = r
                    break
            if reg_obj is not None:
                cols = list(reg_obj.collections())
                print(f"[INFO]: Found {len(cols)} collections in {getattr(reg_obj, 'name', registry_name)} (registries API)")
                return cols
        except Exception as e:
            print(f"[WARN]: registries API unavailable: {e}")
        return None

    @staticmethod
    def _try_list_collections_via_artifact_type(api, entity: str | None, registry_entity: str | None, registry_project: str | None, registry_name: str | None):
        proj_candidates: list[str] = []
        if registry_entity and registry_name:
            proj_candidates.append(f"{registry_entity}/registry/{registry_name}")
            if registry_project:
                proj_candidates.append(f"{registry_entity}/{registry_project}")
        if entity and registry_name:
            proj_candidates.append(f"{entity}/registry/{registry_name}")
            proj_candidates.append(f"{entity}-org/registry/{registry_name}")
            if registry_project:
                proj_candidates.append(f"{entity}/{registry_project}")
                proj_candidates.append(f"{entity}-org/{registry_project}")
        if registry_name:
            proj_candidates.append(f"registry/{registry_name}")
        if registry_project:
            proj_candidates.append(registry_project)
        last_err: Exception | None = None
        for proj in proj_candidates:
            try:
                art_type = api.artifact_type(proj, REGISTRY_NAME)
                cols = list(art_type.collections())
                print(f"[INFO]: Found {len(cols)} collections in {proj} (type={REGISTRY_NAME})")
                return cols
            except Exception as e:
                last_err = e
                continue
        print(f"[WARN]: Registry not found via artifact_type (candidates={proj_candidates}). Last error: {last_err}")
        return None

    @staticmethod
    def _process_collections_parallel(collections, out_txt: Path, processed_rel_set: set[str], processed_stem_set: set[str], output_dir: Path, cfg: dict) -> None:
        synced = 0
        to_download: list[tuple[str, Path]] = []
        # progress for enumerating collections
        total_cols = None
        try:
            total_cols = len(collections)  # collections could be list already
        except Exception:
            total_cols = None
        print(f"[INFO]: Processing {total_cols} collections")
        for coll in _progress_iter(collections, total=total_cols, desc="Sync (collections)"):
            try:
                encoded_name = coll.name
                rel = encoded_name.replace("__", "/")
            except Exception:
                continue
            if rel not in processed_rel_set:
                _append_with_lock(out_txt, rel + "\n")
                processed_rel_set.add(rel)
                processed_stem_set.add(Path(rel).stem)
                synced += 1
            target_npz = (output_dir / rel).with_suffix(".npz")
            if not target_npz.exists():
                to_download.append((rel, target_npz))
        downloaded = 0
        if to_download:
            import multiprocessing as _mp
            pool = _mp.Pool(processes=max(1, int(getattr(args_cli, "sync_workers", 8))))
            results = [
                pool.apply_async(WandbRegistry.download_with_cfg, args=(rel, dst, cfg))
                for rel, dst in to_download
            ]
            pool.close()
            # progress for downloads
            print(f"[INFO]: Downloading {len(results)} npz file(s)")
            for (rel, _), res in _progress_iter(list(zip(to_download, results)), total=len(results), desc="Sync (registry download)"):
                ok = False
                try:
                    ok = bool(res.get())
                except Exception:
                    ok = False
                if ok:
                    downloaded += 1
            pool.join()
        print(f"[INFO]: Synced {synced} entries from wandb to {out_txt}")
        print(f"[INFO]: Downloaded {downloaded} missing npz file(s) into {output_dir}")

    @staticmethod
    def _fallback_sync_via_runs(api, out_txt: Path, processed_rel_set: set[str], processed_stem_set: set[str], output_dir: Path, cfg: dict) -> None:
        run_project = cfg.get("run_project", PROJECT_NAME)
        entity = cfg.get("entity")
        registry_entity = cfg.get("registry_entity")
        run_proj_candidates: list[str] = []
        probe_entities: list[str] = []
        if entity:
            probe_entities.append(entity)
            probe_entities.append(f"{entity}-org")
        if registry_entity:
            probe_entities.append(registry_entity)
        _seen = set()
        probe_entities = [e for e in probe_entities if e and not (e in _seen or _seen.add(e))]
        for ent in probe_entities:
            run_proj_candidates.append(f"{ent}/{run_project}")
        run_proj_candidates.append(run_project)
        runs = []
        last_run_err: Exception | None = None
        last_run_project = None
        for proj in run_proj_candidates:
            try:
                runs = api.runs(proj)
                last_run_project = proj
                break
            except Exception as e:
                last_run_err = e
                continue
        if not runs:
            raise RuntimeError(
                f"Could not sync from registry nor runs. Tried runs={run_proj_candidates}. "
                f"Last error: {last_run_err}"
            )
        print(f"[INFO]: Found {len(runs)} runs in {last_run_project}")
        _synced = 0
        to_download: list[tuple[str, Path]] = []
        for run in _progress_iter(runs, total=len(runs), desc="Sync (runs enumerate)"):
            try:
                rel = run.config.get("rel_path", None)
            except Exception:
                rel = None
            if isinstance(rel, str) and rel:
                if rel not in processed_rel_set:
                    _append_with_lock(out_txt, rel + "\n")
                    processed_rel_set.add(rel)
                    processed_stem_set.add(Path(rel).stem)
                    _synced += 1
                target_npz = (output_dir / rel).with_suffix(".npz")
                if not target_npz.exists():
                    to_download.append((rel, target_npz))
        _downloaded = 0
        if to_download:
            import multiprocessing as _mp
            pool = _mp.Pool(processes=max(1, int(getattr(args_cli, "sync_workers", 8))))
            results = [
                pool.apply_async(WandbRegistry.download_with_cfg, args=(rel, dst, cfg))
                for rel, dst in to_download
            ]
            pool.close()
            for (rel, _), res in _progress_iter(list(zip(to_download, results)), total=len(results), desc="Sync (runs download)"):
                ok = False
                try:
                    ok = bool(res.get())
                except Exception:
                    ok = False
                if ok:
                    _downloaded += 1
            pool.join()
        print(f"[INFO]: Synced {_synced} entries from wandb runs to {out_txt}")
        print(f"[INFO]: Downloaded {_downloaded} missing npz file(s) into {output_dir}")


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
        self.wandb = WandbRegistry(args.wandb_entity, args.wandb_registry_project, args.wandb_registry_entity)

    def discover_inputs(self) -> list[Path]:
        files: list[Path] = []
        if self.input_dir.is_file():
            assert self.input_dir.suffix.lower() in {".npy", ".npz"}
            files.append(self.input_dir)
        else:
            if self.args.recursive:
                for root, _, fnames in os.walk(self.input_dir):
                    root_path = Path(root)
                    for fn in fnames:
                        if fn.endswith(".npy") or fn.endswith(".npz"):
                            files.append(root_path / fn)
            else:
                for fn in os.listdir(self.input_dir):
                    if fn.endswith(".npy") or fn.endswith(".npz"):
                        files.append(self.input_dir / fn)
        files = sorted(files)
        assert len(files) > 0, f"No .npy/.npz files found in {self.input_dir}"
        print(f"[INFO]: Found {len(files)} PHUMA file(s) in {self.input_dir} (recursive={self.args.recursive})")
        return files

    def run(self, sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]) -> None:
        self.processed.load()
        if self.args.sync_from_wandb:
            self.wandb.sync(self.processed, self.output_dir)
        if self.args.verify_uploaded:
            cfg = self.wandb.serialize()
            missing: list[str] = []
            rels_sorted = sorted(self.processed.processed_rel_set)
            if len(rels_sorted) > 0:
                pool = mp.Pool(processes=max(1, int(getattr(args_cli, "sync_workers", 8))))
                results = [pool.apply_async(_verify_exists_task, args=(rel, cfg)) for rel in rels_sorted]
                pool.close()
                for res in _progress_iter(results, total=len(results), desc="Verifying on WandB"):
                    try:
                        rel, ok = res.get()
                    except Exception:
                        rel, ok = ("", False)
                    if rel and not ok:
                        missing.append(rel)
                pool.join()
            missing_path = (self.processed.path.parent / self.args.missing_list_name)
            try:
                with open(missing_path, "w", encoding="utf-8") as f:
                    for rel in missing:
                        f.write(rel + "\n")
                print(f"[INFO]: Verified processed entries. Missing in WandB: {len(missing)}. Saved to {missing_path}")
            except Exception as e:
                print(f"[WARN]: Failed to write missing list: {e}")
        exporter = MotionExporter(sim, scene, joint_names, self.output_dir, self.processed)
        files = self.discover_inputs()

        pool = mp.Pool(processes=max(1, int(self.args.upload_workers)))
        pending: list[tuple[str, "mp.pool.ApplyResult"]] = []  # type: ignore[name-defined]
        for f in files:
            rel = f.relative_to(self.base_root).as_posix()
            if self.processed.contains(rel):
                continue
            saved = exporter.export(f, rel)
            if saved is not None:
                pending.append((rel, self.wandb.upload_async(pool, saved, rel)))
        pool.close()
        uploaded = 0
        for rel, res in pending:
            break
        # Progress bar for uploads
        for rel, res in _progress_iter(pending, total=len(pending), desc="Uploading to WandB"):
            ok = False
            try:
                ok = bool(res.get())
            except Exception as e:
                print(f"[WARN]: Upload failed for {rel}: {e}")
            if ok:
                uploaded += 1
        pool.join()
        print(f"[INFO]: Saved processed list to {self.processed.path}")
        print(f"[INFO]: Successfully uploaded {uploaded} item(s).")


import os
import multiprocessing as mp
import argparse
import numpy as np
from pathlib import Path
from isaaclab.app import AppLauncher
import tempfile
import shutil
import os as _os

# Suppress W&B console noise
_os.environ.setdefault("WANDB_SILENT", "true")  # set to true to suppress wandb console output
# Suppress W&B console noise (valid values for WANDB_CONSOLE: auto/off/wrap/redirect/wrap_raw/wrap_emu)
_os.environ.setdefault("WANDB_CONSOLE", "off")


try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:
    _tqdm = None


def _progress_iter(iterable, total: int | None = None, desc: str | None = None):
    if _tqdm is None:
        return iterable
    kwargs = {}
    if total is not None:
        kwargs["total"] = total
    if desc is not None:
        kwargs["desc"] = desc
    return _tqdm(iterable, **kwargs)


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
    help="Verify entries in processed_motions.txt exist in WandB registry (writes a missing list).",
)
parser.add_argument(
    "--missing_list_name",
    type=str,
    default="missing_in_wandb.txt",
    help="Filename to write motions not found in WandB registry (relative to input root). Default: missing_in_wandb.txt",
)
parser.add_argument(
    "--sync_workers",
    type=int,
    default=8,
    help="Number of parallel workers to download missing files during sync.",
)
parser.add_argument(
    "--no-verify-uploaded",
    dest="verify_uploaded",
    action="store_false",
    help="Disable verification step against WandB registry.",
)
parser.set_defaults(verify_uploaded=True)

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


def _uploader_task(npz_path: str, rel_entry: str, registry_cfg: dict) -> bool:
    """Upload a saved npz to wandb as an artifact. Runs in a worker process."""
    try:
        return WandbRegistry.upload_with_cfg(npz_path, rel_entry, registry_cfg)
    except Exception as e:
        print(f"[WARN][upload] Failed to log to wandb for {rel_entry}: {e}")
        return False


def _verify_exists_task(rel_entry: str, registry_cfg: dict) -> tuple[str, bool]:
    """Worker task: check if an artifact exists for rel_entry in WandB registry."""
    try:
        ok = WandbRegistry.artifact_exists_with_cfg(rel_entry, registry_cfg)
        return rel_entry, bool(ok)
    except Exception:
        return rel_entry, False


def _append_with_lock(file_path: Path, text: str) -> None:
    """Append a line to file with an exclusive lock (best-effort, POSIX)."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        try:
            import fcntl  # POSIX only
            fcntl.flock(f, fcntl.LOCK_EX)
        except Exception:
            pass
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
        try:
            import fcntl  # re-import inside scope for mypy
            fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:
            pass


def _sync_processed_from_wandb(
    out_txt: Path, processed_rel_set: set[str], processed_stem_set: set[str], output_dir: Path, registry_cfg: dict | None = None
) -> None:
    """Sync processed list and local files from WandB REGISTRY (artifact collections).
    If an entry exists on WandB but the local output file is missing, download it to output_dir/rel_path."""
    try:
        cfg = registry_cfg or {
            "entity": args_cli.wandb_entity,
            "registry_entity": args_cli.wandb_registry_entity,
            "registry_project": args_cli.wandb_registry_project,
            "run_project": PROJECT_NAME,
        }
        return WandbRegistry.sync_runs_with_cfg(out_txt, processed_rel_set, processed_stem_set, output_dir, cfg)
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
