"""Batch convert PHUMA motions (.npy/.npz) to npz and upload/link to Weights & Biases.

Overview
- Discovers input files (.npy/.npz) in a directory (optionally recursive) or a single file.
- Replays motion in Isaac/IsaacLab to export per-sample npz preserving relative folder structure.
- Uploads exported npz artifacts to a WandB registry project and links them there.
- Supports Wandb entity vs organization handling by using a non-prefixed canonical registry name.
- Supports saving exported npz files under '--output_dir' mirroring input structure.
- Supports parallelism by increasing '--upload_workers', '--sync_workers' to alleviate network bottlenecks.
- Automatically sanitize input file names from special characters.

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
- Supports startup synchronization with WandB to hydrate processed list and download locally-missing npz.
- Supports verification of uploaded entries against WandB registry (writes a remotely-missing list).
- Final check of any missing entries against the input dataset and re-process them.

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
                            sanitized = sanitize_rel_path(entry)
                            if sanitized != entry:
                                print(f"[WARN]: Sanitized entry from processed list: {entry} -> {sanitized}")
                            self.processed_rel_set.add(sanitized)
                            self.processed_stem_set.add(Path(sanitized).stem)
                print(f"[INFO]: Loaded {len(self.processed_rel_set)} processed entries from {self.path}")
            except Exception as e:
                print(f"[WARN]: Failed to read processed list {self.path}: {e}")

    def append(self, rel_entry: str) -> None:
        sanitized = sanitize_rel_path(rel_entry)
        if sanitized not in self.processed_rel_set:
            _append_with_lock(self.path, sanitized + "\n")
            self.processed_rel_set.add(sanitized)
            self.processed_stem_set.add(Path(sanitized).stem)

    def contains(self, rel_entry: str) -> bool:
        sanitized = sanitize_rel_path(rel_entry)
        return sanitized in self.processed_rel_set or Path(sanitized).stem in self.processed_stem_set

    def remove(self, rel_entry: str) -> None:
        """Remove an entry from processed list (both file and in-memory sets)."""
        sanitized = sanitize_rel_path(rel_entry)
        if sanitized in self.processed_rel_set:
            try:
                _rewrite_excluding_with_lock(self.path, {sanitized})
            except Exception as e:
                print(f"[WARN]: Failed to rewrite processed list when removing {sanitized}: {e}")
            self.processed_rel_set.discard(sanitized)
            self.processed_stem_set.discard(Path(sanitized).stem)

    def backup_and_remove(self, rels_to_remove: set[str]) -> Path | None:
        """Backup current processed list and write a new one with given entries removed.
        Returns backup file path if backup created, else None."""
        if not self.path.exists() or not rels_to_remove:
            return None
        try:
            from datetime import datetime as _dt
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.path.with_name(f"{self.path.stem}.bak.{ts}{self.path.suffix}")
            self.path.rename(backup_path)
            remaining = [rel for rel in sorted(self.processed_rel_set) if rel not in rels_to_remove]
            with open(self.path, "w", encoding="utf-8") as f:
                for rel in remaining:
                    f.write(rel + "\n")
            # refresh in-memory sets
            self.processed_rel_set = set(remaining)
            self.processed_stem_set = {Path(r).stem for r in remaining}
            print(f"[INFO]: Backed up processed list to {backup_path} and removed {len(rels_to_remove)} entries.")
            return backup_path
        except Exception as e:
            print(f"[WARN]: Failed to backup/remove processed list entries: {e}")
            return None


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
        # Link artifact into registry collection via artifact_type(...).collection(...).link_artifact(...)
        try:
            import wandb  # local import to avoid worker import cycles
            api = wandb.Api()
            registry_proj_path = f"{registry_entity}/registry/{registry_name}" if registry_entity else f"registry/{registry_name}"
            art_type = api.artifact_type(registry_proj_path, REGISTRY_NAME)
            coll = art_type.collection(collection)
            link_method = getattr(coll, "link_artifact", None)
            if not callable(link_method):
                raise RuntimeError("ArtifactCollection.link_artifact is unavailable in this wandb version")
            link_method(logged_artifact)  # type: ignore[misc]
            print(f"[INFO][upload] Motion linked via collection.link_artifact: {registry_proj_path}/{collection}")
        except Exception as e1:
            # Fallback to linking into a normal project bucket path to avoid registry path validation
            try:
                target_bucket_path = (
                    f"{registry_entity}/{cfg.get('registry_project', registry_name)}/{collection}"
                    if registry_entity
                    else f"{cfg.get('registry_project', registry_name)}/{collection}"
                )
                logged_artifact.link(target_bucket_path)
                print(f"[INFO][upload] Motion linked via artifact.link to bucket: {target_bucket_path}")
            except Exception as e2:  # noqa: BLE001
                print(f"[WARN][upload] Failed to link artifact into registry/bucket. collection.link_artifact error={e1}; artifact.link error={e2}")
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
    def list_all_rel_entries_with_cfg(cfg: dict) -> set[str]:
        """Return a set of rel_path strings for all motions available in the WandB registry or fallback sources.
        Prefers the Registries API; falls back to artifact_type; finally falls back to runs config 'rel_path'."""
        import wandb
        api = wandb.Api()
        registry_project = cfg.get("registry_project")
        registry_name = cfg.get("registry_name", registry_project)
        entity = cfg.get("entity")
        registry_entity = cfg.get("registry_entity")
        run_project = cfg.get("run_project", PROJECT_NAME)
        # 1) registries API
        cols = WandbRegistry._try_list_collections_via_registries(api, registry_name, registry_entity)
        if cols is not None:
            rels: set[str] = set()
            try:
                total_cols = len(cols)
            except Exception:
                total_cols = None
            for coll in _progress_iter(cols, total=total_cols, desc="Enumerate registry (collections)"):
                try:
                    rels.add(coll.name.replace("__", "/"))
                except Exception:
                    continue
            return rels
        # 2) artifact_type fallback
        cols = WandbRegistry._try_list_collections_via_artifact_type(api, entity, registry_entity, registry_project, registry_name)
        if cols is not None:
            rels = set()
            try:
                total_cols = len(cols)
            except Exception:
                total_cols = None
            for coll in _progress_iter(cols, total=total_cols, desc="Enumerate artifact_type (collections)"):
                try:
                    rels.add(coll.name.replace("__", "/"))
                except Exception:
                    continue
            return rels
        # 3) runs fallback
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
        for proj in run_proj_candidates:
            try:
                runs = api.runs(proj)
                break
            except Exception:
                continue
        rels = set()
        if runs:
            for run in _progress_iter(runs, total=len(runs), desc="Enumerate runs (fallback)"):
                try:
                    rel = run.config.get("rel_path", None)
                except Exception:
                    rel = None
                if isinstance(rel, str) and rel:
                    rels.add(rel)
        return rels

    @staticmethod
    def sync_runs_with_cfg(
        out_txt: Path, processed_rel_set: set[str], processed_stem_set: set[str], output_dir: Path, cfg: dict
    ) -> None:
        import wandb
        api = wandb.Api()
        print("[INFO]: Syncing from WandB registry (collections) ...")
        print(f"[INFO]: cfg: {cfg}")
        print("[INFO]: Trying to list collections via registries API, this may take a few minutes...")
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
        available_rels = WandbRegistry.list_all_rel_entries_with_cfg(cfg)
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
        for rel, res in _progress_iter(pending, total=len(pending), desc=desc):
            ok = False
            try:
                ok = bool(res.get())
            except Exception as e:
                print(f"[WARN]: Upload failed for {rel}: {e}")
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
        available_rels = WandbRegistry.list_all_rel_entries_with_cfg(cfg)
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
            self.wandb.sync(self.processed, self.output_dir)
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
import multiprocessing as mp
import argparse
import numpy as np
from pathlib import Path
from isaaclab.app import AppLauncher
import tempfile
import shutil
import os as _os
import re as _re

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


def _rewrite_excluding_with_lock(file_path: Path, rels_to_remove: set[str]) -> None:
    """Rewrite file excluding given entries with an exclusive lock (best-effort, POSIX)."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[str] = []
    try:
        with open(file_path, "r", encoding="utf-8") as rf:
            existing = [line.rstrip("\n") for line in rf if line.strip()]
    except FileNotFoundError:
        existing = []
    remaining = [line for line in existing if line not in rels_to_remove]
    with open(file_path, "w", encoding="utf-8") as f:
        try:
            import fcntl  # POSIX only
            fcntl.flock(f, fcntl.LOCK_EX)
        except Exception:
            pass
        for line in remaining:
            f.write(line + "\n")
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


def _sanitize_basename(name: str) -> str:
    """Sanitize a filename (basename only) to be WandB-safe and avoid '__' sequences.
    - Remove '+' and '?' explicitly.
    - Replace any other disallowed characters with '_'.
    - Collapse consecutive underscores to a single '_' to avoid '__' inside filenames.
    """
    # Remove explicit symbols
    name = name.replace("+", "").replace("?", "")
    # Replace disallowed chars with underscore (allow alnum, '-', '_', and '.')
    name = "".join(ch if (_re.match(r"[A-Za-z0-9._-]", ch)) else "_" for ch in name)
    # Collapse multiple underscores
    name = _re.sub(r"_+", "_", name)
    return name


def sanitize_rel_path(rel_path: str) -> str:
    """Sanitize only the basename of a relative path string."""
    p = Path(rel_path)
    new_name = _sanitize_basename(p.name)
    if new_name == p.name:
        return rel_path
    return str(p.with_name(new_name))


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
