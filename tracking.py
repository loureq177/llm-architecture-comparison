"""Hugging Face: live training curves (Trackio) and storage of the trained models.

Everything is on when HF_TOKEN (a "Write" token from huggingface.co/settings/tokens)
is set, off otherwise. In the token owner's account:

  * Space  <user>/llm-architecture-comparison-curves (public, free static Space):
    dashboard of the training curves, one run "<preset>/<model>" per model, refreshed
    every SYNC_EVERY seconds. Trackio logs locally; the dashboard reads a snapshot.
  * model  <user>/llm-architecture-comparison (private): weights, training logs and
    results, at the same paths as here (checkpoints/<preset>/, logs/, results/).

    python tracking.py upload --preset helios --models deltablock     # where it was trained
    python tracking.py download --preset helios --models deltablock   # anywhere else
"""

import os
import threading
import time

from config import cfg

PROJECT = "llm-architecture-comparison"
KINDS = ["transformer", "diffusion", "mamba", "deltablock"]
SYNC_EVERY = 600
_run = None
_sync = {"last": 0.0, "deployed": False, "thread": None}


def enabled():
    return bool(os.environ.get("HF_TOKEN"))


def _user():
    from huggingface_hub import whoami

    return whoami()["name"]


def _warn(what, e):
    print(f"[tracking] {what} failed ({type(e).__name__}: {e}) - training goes on without it", flush=True)


def _config():
    """Run configuration as flat scalars (the dashboard stores it as Parquet)."""
    out = {}
    for k, v in vars(cfg).items():
        if k.startswith("_") or k == "root_dir":
            continue
        out[k] = v if v is None or isinstance(v, (bool, int, float, str)) else str(v)
    return out


def start(key):
    global _run
    if not enabled():
        return
    try:
        import trackio

        _run = trackio.init(project=PROJECT, name=f"{cfg.run_name}/{key}", resume="allow", embed=False,
                            config=_config())
    except Exception as e:
        _warn("live dashboard", e)


def _publish():
    """Pushes a snapshot of the local Trackio project to the static dashboard Space."""
    try:
        import trackio
        from trackio.bucket_storage import upload_project_to_bucket_for_static

        name = f"{_user()}/{PROJECT}-curves"
        if not _sync["deployed"]:  # first time: Space, bucket and data
            trackio.sync(PROJECT, space_id=name, bucket_id=f"{name}-data", sdk="static", force=True)
            _sync["deployed"] = True
            print(f"[tracking] live curves: https://huggingface.co/spaces/{name}", flush=True)
        else:
            upload_project_to_bucket_for_static(PROJECT, f"{name}-data")
    except Exception as e:
        _warn("dashboard refresh", e)


def _refresh(force=False):
    t = _sync["thread"]
    if t is not None and t.is_alive():
        if not force:
            return
        t.join()
    if force or time.time() - _sync["last"] >= SYNC_EVERY:
        _sync["last"] = time.time()
        _sync["thread"] = threading.Thread(target=_publish, daemon=True)
        _sync["thread"].start()
        if force:
            _sync["thread"].join()


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}/"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = v
    return out


def log(record, prefix=""):
    if _run is None:
        return
    try:
        import trackio

        r = dict(record)
        step = r.pop("step")
        trackio.log(_flat(r, prefix), step=step)
    except Exception as e:
        _warn("logging", e)
    _refresh()


def finish():
    global _run
    if _run is None:
        return
    try:
        import trackio

        trackio.finish()
    except Exception as e:
        _warn("finish", e)
    _run = None
    _refresh(force=True)


def _files(key):
    """Files of a trained model, relative to cfg.root_dir: weights, training log, results."""
    paths = [os.path.join("checkpoints", cfg.run_name, f"{key}.pt"),
             os.path.join("logs", cfg.run_name, f"{key}.jsonl")]
    paths += [os.path.join("results", cfg.run_name, f) for f in ("metrics.json", "samples.pt", "samples.txt")]
    return [p for p in paths if os.path.isfile(os.path.join(cfg.root_dir, p))]


def upload(keys):
    from huggingface_hub import CommitOperationAdd, HfApi

    api, repo = HfApi(), f"{_user()}/{PROJECT}"
    api.create_repo(repo, private=True, exist_ok=True)
    for key in keys:
        files = _files(key)
        if not files or not files[0].endswith(".pt"):
            print(f"[{key}] no trained model in {cfg.checkpoints_dir} - nothing to upload")
            continue
        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=os.path.join(cfg.root_dir, p)) for p in files]
        api.create_commit(repo, ops, commit_message=f"{cfg.run_name}/{key}")
        print(f"[{key}] uploaded to huggingface.co/{repo}: " + ", ".join(files), flush=True)


def download(keys):
    from huggingface_hub import snapshot_download

    patterns = [f"checkpoints/{cfg.run_name}/{k}.pt" for k in keys]
    patterns += [f"logs/{cfg.run_name}/{k}.jsonl" for k in keys] + [f"results/{cfg.run_name}/*"]
    snapshot_download(f"{_user()}/{PROJECT}", allow_patterns=patterns, local_dir=cfg.root_dir)
    print(f"downloaded to {cfg.root_dir}: " + ", ".join(patterns))


if __name__ == "__main__":
    import argparse

    from config import PRESETS

    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["upload", "download"])
    p.add_argument("--preset", choices=PRESETS, required=True)
    p.add_argument("--models", nargs="+", choices=KINDS, default=KINDS)
    args = p.parse_args()
    if not enabled():
        raise SystemExit("set HF_TOKEN (a Write token from https://huggingface.co/settings/tokens)")
    cfg.apply_preset(args.preset)
    (upload if args.action == "upload" else download)(args.models)
