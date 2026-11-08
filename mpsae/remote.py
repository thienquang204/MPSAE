"""Native campaign with private Hugging Face backup and W&B project cvpr.

Only this explicitly invoked entry point enables automatic remote writes.
Scientific identities and the standalone local CLI remain unchanged.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import uuid

from .benchmark.config import load_campaign
from .benchmark.run import campaign_paths
from .cli import load_dotenv
from .io import atomic_json, exclusive_lock, read_json
from .logs import METRIC_SINK, get_logger, setup_logging

DEFAULT_PROJECT = "cvpr"
REPOSITORY_NAME = "MPSAE"
BACKUP_FOLDERS = ("embeddings", "benchmarks", "runs", "evaluations", "campaigns", "reports")
SUFFIXES = (".npz", ".json", ".jsonl", ".pt", ".csv", ".md", ".png", ".pdf", ".caption.txt")


def configure_tracking():
    """Accept the user's token spelling; keep the SDK's standard key working."""
    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_TOKEN"):
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_TOKEN"]
    return os.environ.get("WANDB_PROJECT") or DEFAULT_PROJECT


class RemoteSession:
    def __init__(self, campaign, workspace, *, smoke=False):
        self.campaign, self.smoke = campaign, smoke
        self.paths = campaign_paths(campaign, workspace, smoke=smoke)
        self.root = self.paths["workspace"].resolve()
        self.project = configure_tracking()
        self.api = self.run = None
        self.repo = None
        self.wandb_module = None
        self.wandb_failed = False
        self.defined_metrics = set()
        self.directory = self.paths["state"] / "services"
        self.status = {"repository": None, "private": True, "wandb_project": self.project,
                       "smoke": smoke, "hf_sync_complete": None, "wandb_complete": None,
                       "warnings": []}

    def save(self):
        atomic_json(self.directory / "status.json", self.status)

    def warn(self, operation, error):
        message = str(error)
        for key in ("HF_TOKEN", "WANDB_TOKEN", "WANDB_API_KEY"):
            token = os.environ.get(key)
            if token:
                message = message.replace(token, "[REDACTED]")
        detail = {"operation": operation, "error": f"{type(error).__name__}: {message}"}
        self.status["warnings"].append(detail)
        get_logger("remote").warning("%s: %s", operation, detail["error"])
        self.save()

    def ensure_repository(self):
        # Recreate a deleted repo on every invocation; do not trust a local
        # uploaded=true receipt that may refer to an earlier repository.
        self.api.create_repo(repo_id=self.repo, repo_type="dataset", private=True, exist_ok=True)
        if not self.api.repo_info(repo_id=self.repo, repo_type="dataset").private:
            raise RuntimeError(f"{self.repo} already exists publicly; make it private before backing up ImageNet-derived artifacts")

    def start(self):
        if self.smoke:
            get_logger("remote").info("Synthetic rehearsal: remote writes disabled")
            return
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("Set HF_TOKEN in .env with read access to ImageNet and write access for the private MPSAE repository")
        from huggingface_hub import HfApi

        self.api = HfApi(token=token)
        account = self.api.whoami().get("name")
        if not account or "/" in account or account in (".", ".."):
            raise RuntimeError("HF_TOKEN did not resolve to a personal Hugging Face account")
        self.repo = f"{account}/{REPOSITORY_NAME}"
        self.ensure_repository()
        self.status["repository"] = self.repo
        get_logger("remote").info("Private backup: https://huggingface.co/datasets/%s", self.repo)
        if os.environ.get("WANDB_API_KEY"):
            import wandb

            self.wandb_module = wandb
            identity_file = self.directory / "wandb.json"
            identity = read_json(identity_file) or {}
            if identity.get("project") != self.project:
                identity = {"project": self.project, "id": uuid.uuid4().hex[:16]}
                atomic_json(identity_file, identity)
            self.directory.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(project=self.project, id=identity["id"], resume="allow",
                                  name=f"MPSAE-{self.campaign.name}", job_type="campaign",
                                  config=self.campaign.plan(), dir=str(self.directory), mode="online",
                                  save_code=False, settings=wandb.Settings(init_timeout=60, console="off"))
            if self.run is None:
                raise RuntimeError("W&B did not start an online run; check WANDB_TOKEN")
            self.status["wandb_run_id"] = identity["id"]
        else:
            get_logger("remote").info("No WANDB_TOKEN configured; HF backup remains enabled")
        self.save()

    def upload(self, folder):
        folder = Path(folder).resolve()
        if not folder.is_relative_to(self.root) or folder == self.root:
            raise ValueError("backup folder must be a generated artifact folder inside the workspace")
        relative = folder.relative_to(self.root)
        if relative.parts[0] not in BACKUP_FOLDERS:
            raise ValueError("refusing to upload a folder outside the artifact allowlist")
        if not folder.is_dir():
            return
        files = []
        for path in folder.rglob("*"):
            if (not path.is_file() or path.is_symlink() or path.resolve() != path.absolute()
                    or any(part.startswith(".") or part in ("wandb", "logs", "cache", "codes")
                           for part in path.relative_to(folder).parts)
                    or not path.name.endswith(SUFFIXES)):
                continue
            files.append(path.relative_to(folder).as_posix())
        # Explicit file lists prevent cache files, raw images, env files and
        # incomplete writes from entering the backup. Commits stay bounded.
        files.sort()
        for first in range(0, len(files), 100):
            self.api.upload_folder(repo_id=self.repo, repo_type="dataset", folder_path=str(folder),
                                   path_in_repo=relative.as_posix(), allow_patterns=files[first:first + 100],
                                   commit_message=f"MPSAE {self.campaign.name}: {relative.as_posix()}")

    def on_task(self, name, result):
        if self.smoke:
            return
        # Protect expensive artifacts immediately. Scalar evaluations and final
        # plots are synchronized in finish, including on a failed/interrupted run.
        folder = None
        if name.startswith("imagenet/embed/"):
            folder = self.root / "embeddings" / name.rsplit("/", 1)[1]
        elif name.startswith("source/"):
            folder = self.root / "runs" / name.removeprefix("source/")
        elif name.startswith("features/") and result is not None:
            path = getattr(result.gallery, "path", None)
            if path is not None:
                folder = Path(path).parent
        if folder is not None:
            try:
                self.ensure_repository()
                self.upload(folder)
            except Exception as error:
                self.warn(f"backup {name}", error)
        if self.run is not None and not self.wandb_failed:
            try:
                payload = {"campaign/task": name}
                if isinstance(result, dict) and "metrics" in result:
                    payload.update({f"retrieval/{key}": value for key, value in result["metrics"].items()
                                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                                    and math.isfinite(value)})
                    payload.update({f"retrieval/{key}": result.get(key)
                                    for key in ("dataset", "backbone", "method", "train_k", "eval_k", "seed")})
                self.run.log(payload)
            except Exception as error:
                self.wandb_failed = True
                self.warn("W&B task metrics", error)

    def training_metrics(self, stage, record, step):
        if self.run is None or self.wandb_failed:
            return
        try:
            prefix = f"train/{stage}"
            if prefix not in self.defined_metrics:
                self.run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/step")
                self.defined_metrics.add(prefix)
            values = {f"{prefix}/{key}": value for key, value in record.items()
                      if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)}
            values[f"{prefix}/step"] = step
            self.run.log(values)
        except Exception as error:
            self.wandb_failed = True
            self.warn("W&B training metrics", error)

    def finish(self, *, scientific_success):
        if self.smoke:
            return True
        synced = True
        try:
            self.ensure_repository()
        except Exception as error:
            synced = False
            self.warn("final HF synchronization", error)
        else:
            for folder in BACKUP_FOLDERS:
                try:
                    self.upload(self.root / folder)
                except Exception as error:
                    synced = False
                    self.warn(f"final HF synchronization: {folder}", error)
        self.status["hf_sync_complete"] = synced
        self.status["scientific_success"] = scientific_success
        if self.run is not None:
            try:
                catalog = read_json(self.paths["state"] / "catalog.json") or {}
                records = catalog.get("evaluations", [])
                columns = ["dataset", "backbone", "method", "train_k", "eval_k", "seed", "recall_at_1", "map_at_r"]
                rows = [[record.get(key) if key not in ("recall_at_1", "map_at_r")
                         else record.get("metrics", {}).get(key) for key in columns] for record in records]
                self.run.log({"retrieval/results": self.wandb_module.Table(columns=columns, data=rows)})
                if self.paths["reports"].is_dir():
                    artifact = self.wandb_module.Artifact(f"MPSAE-{self.campaign.plan()['plan_id'][:12]}", type="report")
                    artifact.add_dir(str(self.paths["reports"]))
                    self.run.log_artifact(artifact)
            except Exception as error:
                self.wandb_failed = True
                self.warn("W&B final report", error)
            finally:
                try:
                    self.run.finish(exit_code=0 if scientific_success and not self.wandb_failed else 1)
                except Exception as error:
                    self.wandb_failed = True
                    self.warn("W&B finalization", error)
        self.status["wandb_complete"] = not self.wandb_failed if self.run is not None else None
        self.save()
        if synced:
            try:
                self.upload(self.directory)
            except Exception as error:
                synced = False
                self.status["hf_sync_complete"] = False
                self.warn("backup service status", error)
        return synced and not self.wandb_failed


def main(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--env-file", default=".env")
    initial, _ = bootstrap.parse_known_args(argv)
    load_dotenv(initial.env_file)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "sync", "check"))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--config", default="configs/thesis.yaml")
    parser.add_argument("--workspace", default=os.environ.get("MPSAE_WORKSPACE", "work"))
    parser.add_argument("--device", default=os.environ.get("MPSAE_DEVICE", "cuda:0"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("MPSAE_WORKERS", "16")))
    parser.add_argument("--embed-batch-size", type=int, default=int(os.environ.get("MPSAE_EMBED_BATCH", "256")))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_campaign(args.config)
    session = RemoteSession(campaign, args.workspace, smoke=args.smoke)
    setup_logging(session.root / "logs", name="all-in-one")
    # Keep simultaneous all-in-one invocations from resuming the same W&B run
    # or uploading files while another invocation is replacing its artifacts.
    with exclusive_lock(session.directory / "remote.lock", timeout=0):
        return execute(session, args)


def execute(session, args):
    try:
        session.start()
    except Exception as error:
        session.warn("remote initialization", error)
        raise
    if args.command == "check":
        if session.run is not None:
            session.run.finish()
        return 0
    sink_token = METRIC_SINK.set(session.training_metrics)
    scientific_success = False
    try:
        if args.command == "run":
            from .benchmark.run import run_campaign

            result = run_campaign(session.campaign, args.workspace, device=args.device, workers=args.workers,
                                  embed_batch_size=args.embed_batch_size, smoke=args.smoke, observer=session)
            scientific_success = bool(result.get("complete"))
        else:
            manifest = read_json(session.paths["reports"] / "manifest.json") or {}
            scientific_success = bool(manifest.get("complete"))
    finally:
        METRIC_SINK.reset(sink_token)
        remote_success = session.finish(scientific_success=scientific_success)
    if not remote_success:
        raise RuntimeError(f"Remote synchronization incomplete; see {session.directory / 'status.json'}. "
                           "Retry bash all_in_one.sh --sync-only with the same config/workspace.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # HTTP exceptions can contain request information; redact token values.
        message = str(error)
        for key in ("HF_TOKEN", "WANDB_TOKEN", "WANDB_API_KEY"):
            if os.environ.get(key):
                message = message.replace(os.environ[key], "[REDACTED]")
        print(f"all-in-one failed: {type(error).__name__}: {message}", file=sys.stderr)
        raise SystemExit(1)
