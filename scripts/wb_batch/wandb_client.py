from __future__ import annotations

from pathlib import Path
import tempfile
import shutil
import multiprocessing as _mp

from .filelock import append_with_lock
from .pool import progress_iter, drain_results

# Artifact type name used in registry
REGISTRY_TYPE_NAME = "Motions"


class WandbRegistry:
    """Encapsulate Weights & Biases interactions (upload, link, sync, list, exists, download)."""

    def __init__(self, entity: str | None, registry_project: str, registry_entity: str | None):
        self.entity = entity
        self.registry_project = registry_project
        self.registry_entity = registry_entity
        # Non-prefixed canonical registry name (UI: orgs/<org>/registry/<Name>)
        if isinstance(registry_project, str) and registry_project.startswith("wandb-registry-"):
            self.registry_name = registry_project[len("wandb-registry-") :]
        else:
            self.registry_name = registry_project
        # Common project name used to create runs (caller should override via cfg if needed)
        self.run_project = "csv_to_npz"

    def serialize(self) -> dict:
        return {
            "entity": self.entity,
            "registry_project": self.registry_project,
            "registry_entity": self.registry_entity,
            "registry_name": self.registry_name,
            "run_project": self.run_project,
        }

    def upload_async(self, pool, npz_path: Path, rel_entry: str):
        cfg = self.serialize()
        return pool.apply_async(WandbRegistry.upload_with_cfg, args=(npz_path.as_posix(), rel_entry, cfg))

    # ---------------------------- static utilities (workers) ----------------------------
    @staticmethod
    def upload_with_cfg(npz_path: str, rel_entry: str, cfg: dict) -> bool:
        import wandb

        collection = rel_entry.replace("/", "__").replace("\\", "__")
        # choose run entity
        entity = cfg.get("entity")
        registry_entity = cfg.get("registry_entity")
        registry_name = cfg.get("registry_name", cfg.get("registry_project"))
        run_project = cfg.get("run_project", "csv_to_npz")
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
            api = wandb.Api()
            registry_proj_path = f"{registry_entity}/registry/{registry_name}" if registry_entity else f"registry/{registry_name}"
            art_type = api.artifact_type(registry_proj_path, REGISTRY_TYPE_NAME)
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
        run_project = cfg.get("run_project", "csv_to_npz")
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
        run_project = cfg.get("run_project", "csv_to_npz")
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

    # ---------------------------- sync/list helpers ----------------------------
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
                art_type = api.artifact_type(proj, REGISTRY_TYPE_NAME)
                cols = list(art_type.collections())
                print(f"[INFO]: Found {len(cols)} collections in {proj} (type={REGISTRY_TYPE_NAME})")
                return cols
            except Exception as e:
                last_err = e
                continue
        print(f"[WARN]: Registry not found via artifact_type (candidates={proj_candidates}). Last error: {last_err}")
        return None

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
        run_project = cfg.get("run_project", "csv_to_npz")
        # 1) registries API
        cols = WandbRegistry._try_list_collections_via_registries(api, registry_name, registry_entity)
        if cols is not None:
            rels: set[str] = set()
            try:
                total_cols = len(cols)
            except Exception:
                total_cols = None
            for coll in progress_iter(cols, total=total_cols, desc="Enumerate registry (collections)"):
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
            for coll in progress_iter(cols, total=total_cols, desc="Enumerate artifact_type (collections)"):
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
            for run in progress_iter(runs, total=len(runs), desc="Enumerate runs (fallback)"):
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
        sync_workers = int(cfg.get("sync_workers", 8))
        # 1) registries API
        collections = WandbRegistry._try_list_collections_via_registries(api, registry_name, registry_entity)
        if collections is not None:
            WandbRegistry._process_collections_parallel(collections, out_txt, processed_rel_set, processed_stem_set, output_dir, cfg, sync_workers)
            return
        # 2) artifact_type probing fallback
        collections = WandbRegistry._try_list_collections_via_artifact_type(api, entity, registry_entity, registry_project, registry_name)
        if collections is not None:
            WandbRegistry._process_collections_parallel(collections, out_txt, processed_rel_set, processed_stem_set, output_dir, cfg, sync_workers)
            return
        # 3) runs fallback
        WandbRegistry._fallback_sync_via_runs(api, out_txt, processed_rel_set, processed_stem_set, output_dir, cfg, sync_workers)

    @staticmethod
    def _process_collections_parallel(collections, out_txt: Path, processed_rel_set: set[str], processed_stem_set: set[str], output_dir: Path, cfg: dict, sync_workers: int) -> None:
        synced = 0
        to_download: list[tuple[str, Path]] = []
        # progress for enumerating collections
        total_cols = None
        try:
            total_cols = len(collections)  # collections could be list already
        except Exception:
            total_cols = None
        print(f"[INFO]: Processing {total_cols} collections")
        for coll in progress_iter(collections, total=total_cols, desc="Sync (collections)"):
            try:
                encoded_name = coll.name
                rel = encoded_name.replace("__", "/")
            except Exception:
                continue
            if rel not in processed_rel_set:
                append_with_lock(out_txt, rel + "\n")
                processed_rel_set.add(rel)
                processed_stem_set.add(Path(rel).stem)
                synced += 1
            target_npz = (output_dir / rel).with_suffix(".npz")
            if not target_npz.exists():
                to_download.append((rel, target_npz))
        downloaded = 0
        if to_download:
            pool = _mp.Pool(processes=max(1, int(sync_workers)))
            results = [
                pool.apply_async(WandbRegistry.download_with_cfg, args=(rel, dst, cfg))
                for rel, dst in to_download
            ]
            pool.close()
            print(f"[INFO]: Downloading {len(results)} npz file(s)")
            for rel, ok in drain_results(list(zip((r for r, _ in to_download), results)), total=len(results), desc="Sync (registry download)"):
                if ok:
                    downloaded += 1
            pool.join()
        print(f"[INFO]: Synced {synced} entries from wandb to {out_txt}")
        print(f"[INFO]: Downloaded {downloaded} missing npz file(s) into {output_dir}")

    @staticmethod
    def _fallback_sync_via_runs(api, out_txt: Path, processed_rel_set: set[str], processed_stem_set: set[str], output_dir: Path, cfg: dict, sync_workers: int) -> None:
        import multiprocessing as _mp

        run_project = cfg.get("run_project", "csv_to_npz")
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
        for run in progress_iter(runs, total=len(runs), desc="Sync (runs enumerate)"):
            try:
                rel = run.config.get("rel_path", None)
            except Exception:
                rel = None
            if isinstance(rel, str) and rel:
                if rel not in processed_rel_set:
                    append_with_lock(out_txt, rel + "\n")
                    processed_rel_set.add(rel)
                    processed_stem_set.add(Path(rel).stem)
                    _synced += 1
                target_npz = (output_dir / rel).with_suffix(".npz")
                if not target_npz.exists():
                    to_download.append((rel, target_npz))
        _downloaded = 0
        if to_download:
            pool = _mp.Pool(processes=max(1, int(sync_workers)))
            results = [
                pool.apply_async(WandbRegistry.download_with_cfg, args=(rel, dst, cfg))
                for rel, dst in to_download
            ]
            pool.close()
            for rel, ok in drain_results(list(zip((r for r, _ in to_download), results)), total=len(results), desc="Sync (runs download)"):
                if ok:
                    _downloaded += 1
            pool.join()
        print(f"[INFO]: Downloaded {_downloaded} missing npz file(s) into {output_dir}")
