"""Push generated route outputs to a Hugging Face dataset repo instead of local disk.

Each route lands in its OWN FOLDER on the repo: <route_name>/<route_name>.json, so a dev
can pull just the routes they want. The dataset artifact never touches local disk — it is
serialized in memory and uploaded. (The per-route STATE checkpoint under synth/state/ is
kept locally regardless: it is the resume ledger, not the deliverable.)

Auth: HF_API_KEY (env or ROOT/.env), reusing engine's .env reader. Requires
huggingface_hub (present in the nanochat venv the pipeline runs under).
"""

import io
import json
import os
import sys
from pathlib import Path

# Standalone (no engine import, so it stays free of the httpx transport dep and runs in
# any env that has huggingface_hub).
ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "zachnorton03/synthetic-pre1930-sft"
HF_REPO_TYPE = "dataset"


def _from_dotenv(name):
    """Read NAME from ROOT/.env (KEY=value or `export KEY=value`), else None."""
    f = ROOT / ".env"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        line = line[len("export "):].strip() if line.startswith("export ") else line
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip('"').strip("'")
    return None


def _token():
    tok = os.environ.get("HF_API_KEY") or _from_dotenv("HF_API_KEY")
    if not tok:
        sys.exit("Set HF_API_KEY (env or ROOT/.env) to push to Hugging Face.")
    return tok


def _api():
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError:
        sys.exit("huggingface_hub not installed in this env. Install it into the venv "
                 "you run under, e.g.  uv pip install --python .venv/bin/python huggingface_hub")
    return HfApi(token=_token())


def push_shard(route_name, rows, repo=HF_REPO):
    """Upload `rows` as a NEW timestamped JSONL shard under unfiltered/<route_name>/ on
    the HF dataset repo (raw generation output; the verify filter reads from there and
    writes verified/<route_name>/). Commits incrementally without re-uploading prior
    shards. Returns the repo path. Creates the repo if missing."""
    from datetime import datetime, timezone
    api = _api()
    api.create_repo(repo_id=repo, repo_type=HF_REPO_TYPE, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path_in_repo = f"unfiltered/{route_name}/part-{stamp}.jsonl"
    data = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")
    api.upload_file(
        path_or_fileobj=io.BytesIO(data),
        path_in_repo=path_in_repo,
        repo_id=repo,
        repo_type=HF_REPO_TYPE,
        commit_message=f"{route_name}: +{len(rows)} rows",
    )
    return f"{repo}/{path_in_repo}"


def write_sharded(prefix, rows, shard_size=2000, repo=HF_REPO):
    """Write `rows` as uniform SHARD_SIZE-row JSONL shards under `prefix`/ on the repo
    (e.g. prefix='filtered/knowledge_qa' -> filtered/knowledge_qa/part-00000.jsonl …),
    replacing anything already under that prefix, in ONE atomic commit. Returns
    (replaced, new_shards, rows)."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    api = _api()
    api.create_repo(repo_id=repo, repo_type=HF_REPO_TYPE, exist_ok=True)
    existing = [f for f in api.list_repo_files(repo_id=repo, repo_type=HF_REPO_TYPE)
                if f.startswith(f"{prefix}/")]
    ops = [CommitOperationDelete(path_in_repo=f) for f in existing]
    n_new = 0
    for i in range(0, len(rows), shard_size):
        chunk = rows[i:i + shard_size]
        body = ("\n".join(json.dumps(r, ensure_ascii=False) for r in chunk) + "\n").encode("utf-8")
        ops.append(CommitOperationAdd(path_in_repo=f"{prefix}/part-{n_new:05d}.jsonl",
                                      path_or_fileobj=body))
        n_new += 1
    api.create_commit(repo_id=repo, repo_type=HF_REPO_TYPE, operations=ops,
                      commit_message=f"write {prefix}: {len(rows)} rows in {n_new} shards")
    return len(existing), n_new, len(rows)


def compact_route(route_name, shard_size=2000, repo=HF_REPO):
    """Rewrite a route's shards to a uniform SHARD_SIZE rows each (last shard holds the
    remainder). Reads every existing <route>/part-*.jsonl in generation order, re-chunks,
    and swaps old->new in ONE atomic commit (adds + deletes together, so an interruption
    can't drop data). New shards are sequentially named part-00000.jsonl … Returns
    (old_shards, new_shards, rows)."""
    from huggingface_hub import hf_hub_download, CommitOperationAdd, CommitOperationDelete
    api = _api()
    old = sorted(f for f in api.list_repo_files(repo_id=repo, repo_type=HF_REPO_TYPE)
                 if f.startswith(f"{route_name}/") and f.endswith((".jsonl", ".json")))
    rows = []
    for f in old:
        local = hf_hub_download(repo_id=repo, repo_type=HF_REPO_TYPE, filename=f, token=_token())
        with open(local, encoding="utf-8") as fh:
            if f.endswith(".json"):
                rows.extend(json.load(fh))
            else:
                rows.extend(json.loads(l) for l in fh if l.strip())
    ops = [CommitOperationDelete(path_in_repo=f) for f in old]
    n_new = 0
    for i in range(0, len(rows), shard_size):
        chunk = rows[i:i + shard_size]
        body = ("\n".join(json.dumps(r, ensure_ascii=False) for r in chunk) + "\n").encode("utf-8")
        ops.append(CommitOperationAdd(path_in_repo=f"{route_name}/part-{n_new:05d}.jsonl",
                                      path_or_fileobj=body))
        n_new += 1
    api.create_commit(repo_id=repo, repo_type=HF_REPO_TYPE, operations=ops,
                      commit_message=f"compact {route_name}: {len(old)} -> {n_new} shards of <= {shard_size}")
    return len(old), n_new, len(rows)


def migrate_route(route_name, drop_keys=("excerpt", "category_moved"), repo=HF_REPO):
    """One-time fixup for shards already pushed under the old schema/format: for every
    <route_name>/part-*.json shard, strip drop_keys from each row, rewrite it as a .jsonl
    shard, and delete the old .json. Returns (shards_migrated, rows_migrated)."""
    from huggingface_hub import hf_hub_download
    api = _api()
    old = sorted(f for f in api.list_repo_files(repo_id=repo, repo_type=HF_REPO_TYPE)
                 if f.startswith(f"{route_name}/") and f.endswith(".json"))
    total = 0
    for f in old:
        local = hf_hub_download(repo_id=repo, repo_type=HF_REPO_TYPE, filename=f,
                                token=_token())
        with open(local, encoding="utf-8") as fh:
            rows = json.load(fh)
        for row in rows:
            for k in drop_keys:
                row.pop(k, None)
        new_path = f[:-len(".json")] + ".jsonl"
        body = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")
        api.upload_file(path_or_fileobj=io.BytesIO(body), path_in_repo=new_path,
                        repo_id=repo, repo_type=HF_REPO_TYPE,
                        commit_message=f"migrate {f} -> jsonl, drop {list(drop_keys)}")
        api.delete_file(path_in_repo=f, repo_id=repo, repo_type=HF_REPO_TYPE,
                        commit_message=f"remove pre-migration {f}")
        total += len(rows)
        print(f"  migrated {f} -> {new_path}  ({len(rows)} rows)")
    return len(old), total


def test_connection(repo=HF_REPO):
    """Validate auth + write access without leaving junk: upload a tiny probe file,
    confirm it lands, then delete it. Prints the outcome and returns True on success."""
    from datetime import datetime
    api = _api()
    api.create_repo(repo_id=repo, repo_type=HF_REPO_TYPE, exist_ok=True)
    probe = f".connection_test/probe_{datetime.now():%Y%m%d_%H%M%S}.json"
    payload = json.dumps({"ok": True, "at": datetime.now().isoformat()}).encode("utf-8")
    api.upload_file(path_or_fileobj=io.BytesIO(payload), path_in_repo=probe,
                    repo_id=repo, repo_type=HF_REPO_TYPE,
                    commit_message="connection test (temporary probe)")
    files = api.list_repo_files(repo_id=repo, repo_type=HF_REPO_TYPE)
    landed = probe in files
    if landed:
        api.delete_file(path_in_repo=probe, repo_id=repo, repo_type=HF_REPO_TYPE,
                        commit_message="remove connection-test probe")
    print(f"HF connection to {repo}: {'OK (probe uploaded, verified, removed)' if landed else 'FAILED'}")
    return landed


if __name__ == "__main__":
    test_connection()
