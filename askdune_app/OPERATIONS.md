# AskDUNE operations runbook

This runbook covers **development operations on `askdunegpvm01`** and a safe checklist for planning production operations. It does **not** document a verified production service manager or authorize changing the live service.

## 1. Hosts, paths, and boundaries

- Development host: `askdunegpvm01.fnal.gov` (account `dune`).
- Repository directory: `/exp/dune/app/users/dune/askdune/dev/askdune-v1.0.0/askdune/askdune_app`.
- Production AskDUNE uses port **8000** with Fermilab SSO. **Do not stop, restart, replace, or bind to port 8000** using this runbook.
- Development server uses port **8001**, bound to `127.0.0.1`, with authentication disabled only for private testing.
- Preserve both existing indexes: stable legacy development ChromaDB under `../test_data/indexing` and experimental parent-child ChromaDB under `../test_data/parent_child_experiment` (paths relative to the repository). Do not delete, overwrite, or rebuild either without a reviewed plan.
- Do not commit `.env`, credentials, ChromaDB data, or local backup files.

## 2. Prepare a development shell

```bash
ssh askdune
cd /exp/dune/app/users/dune/askdune/dev/askdune-v1.0.0/askdune/askdune_app
source .venv/bin/activate
```

Inspect configuration without printing secrets:

```bash
python - <<'PY'
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
for key in ('RETRIEVAL_BACKEND', 'EMBEDDING_MODEL', 'PROD_DB_PATH',
            'INDEXING_DB_PATH', 'UNIFIED_INDEX_PATH'):
    value = os.getenv(key)
    if key.endswith('_PATH') and value:
        print(f'{key}: {Path(value).expanduser().resolve()}')
    else:
        print(f'{key}: {value or "<unset>"}')
PY
```

`PROD_DB_PATH` is still required by current configuration even when using the unified retrieval backend. Verify that the selected index exists and matches its embedding model before starting the app.

## 3. Start the development web server (foreground)

In a dedicated SSH terminal, from the repository root with `.venv` active:

```bash
ENABLE_AUTHENTICATION=false RETRIEVAL_BACKEND=unified PYTHONPATH=. \
  python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Keep this terminal open. In a **second** terminal, connect from your local computer:

```bash
ssh -N -L 8001:127.0.0.1:8001 askdune
```

Open `http://127.0.0.1:8001/` in your local browser. The development instance must **not** be exposed on `0.0.0.0` while authentication is disabled.

To check the development endpoint from the server:

```bash
curl -I http://127.0.0.1:8001/
```

Some applications do not implement `HEAD`; if that returns 405, try `curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8001/`.

## 4. Stop or inspect the development server

If running in the foreground, press **Ctrl+C in the terminal that started port 8001**. Do not stop an unidentified Python or Uvicorn process.

To identify the process listening on the development port:

```bash
ss -ltnp | grep ':8001'
```

If you no longer control the starting terminal, inspect the PID and command line before taking any action. **Never use a broad `pkill uvicorn` or `killall python`**, because these may affect production.

## 5. Indexing: prerequisites and destination checks

The current parent-child scripts are **experimental**, not production jobs. They derive their write destination as a sibling `parent_child_experiment` directory of `INDEXING_DB_PATH`; they do not simply write to `UNIFIED_INDEX_PATH`. The experimental directory must already exist. Confirm the resolved paths agree with your intended test index.

Required precautions:

1. Confirm source access (DocDB or Indico) and the necessary credentials are configured in `.env`.
2. Confirm `EMBEDDING_MODEL=intfloat/e5-small-v2` for these scripts.
3. Resolve `INDEXING_DB_PATH`, its sibling `parent_child_experiment`, `PROD_DB_PATH`, and `UNIFIED_INDEX_PATH`. Make sure the experimental write destination is **not** the production or stable legacy index.
4. Arrange a backup/snapshot and ensure no conflicting indexing writer is running before a real batch.
5. Begin with a small batch and review logs, ChromaDB counts, and retrieval behavior before scaling.

Read-only CLI inspection:

```bash
python scripts/run_docdb_indexing.py --help
python scripts/run_indico_indexing.py --help
```

## 6. Run a small DocDB indexing batch

**Only after the checks in section 5**:

```bash
python scripts/run_docdb_indexing.py --start 1 --limit 5 --batch-size 32
```

`--start` defaults to 1, `--limit` to 5, and `--batch-size` to 32. The pipeline prepares DocDB documents and attachments, embeds smaller children, and tracks active document versions in the DocDB manifest. A repeated run is not a substitute for confirming actual update/version behavior.

## 7. Run a small Indico indexing batch

**Only after the checks in section 5**:

```bash
python scripts/run_indico_indexing.py --start 0 --limit 5 --batch-size 32
```

For one event, inspect `--help` and use `--event-id` with the intended event identifier. The Indico script skips events whose existing children are all marked complete; its completion logic differs from DocDB version tracking.

## 8. Background jobs and logging (development only)

For a long-running indexing test, a persistent `tmux` session is preferable to an unmanaged background process. First verify `tmux` is installed:

```bash
command -v tmux
```

Start a named session with `tmux new -s askdune-index-test`, activate `.venv`, and run the **reviewed** indexing command there. Detach with **Ctrl+B**, then **D**; reconnect using `tmux attach -t askdune-index-test`. Use distinct session names for the web server and indexing, and do not reuse production session names.

For development logging, optionally redirect a command's output to a timestamped file in a private, writable directory (never commit logs, which may contain document text or operational details):

```bash
mkdir -p "$HOME/askdune-dev-logs"
python scripts/run_docdb_indexing.py --start 1 --limit 5 --batch-size 32 \
  > "$HOME/askdune-dev-logs/docdb-test-$(date +%Y%m%d-%H%M%S).log" 2>&1
```

Check the exit code (`echo $?` immediately afterward) and inspect the log before assuming success. Redirecting output alone does not keep a job alive after logout; use a persistent terminal session for that.

## 9. Scheduling and incremental indexing: not yet deployed by this guide

Do **not** install cron entries, systemd units, or automatic full-corpus jobs based solely on the example commands above. Before scheduling, document and validate:

- The exact indexing account, working directory, Python environment, and secret-loading method.
- Separate read/write permissions and confirmed index destination.
- Incremental discovery strategy, deletion/version handling, and rerun idempotency for each source.
- A single-writer policy or lock, runtime limits, retry/backoff, rate limits, and failure notifications.
- Log rotation, disk quotas, backups, rollback, and post-run quality checks.
- How a validated new index becomes available to retrieval without interrupting production.

A scheduler should invoke a reviewed wrapper with absolute paths and a lock, not copy an interactive command blindly.

## 10. Production service: verify before documenting commands

The live SSO-enabled AskDUNE service is separately managed. Its actual startup mechanism (for example, systemd, container, supervisor, or an operations-managed launcher), service name, working directory, restart policy, logs, and deployment approval process **have not been verified for this runbook**.

Before adding production start/stop commands, ask the service owner to establish those facts and record a tested, authorized procedure. In particular, do not run the unauthenticated development command on the public interface or production port.

## 11. Troubleshooting checklist

| Symptom | First safe check |
| --- | --- |
| Dev browser cannot connect | Confirm SSH tunnel, foreground Uvicorn process, and port 8001 listener |
| App refuses to start | Check `.env` presence, `PROD_DB_PATH`, `UNIFIED_INDEX_PATH`, and Python imports; avoid exposing secret values |
| No unified results | Verify index location, expected collections, embedding compatibility, and indexing coverage |
| Indexing refuses to run | Review script `--help`, credentials, `EMBEDDING_MODEL`, and separate experimental destination |
| Unexpectedly slow search | Inspect retrieval logs and model loading; do not assume indexing is the bottleneck |
| Index counts look wrong | Stop further test writes and review source selection, version/completion logic, and logs before rerunning |
| Port already occupied | Identify the exact listening PID; do not terminate production processes |

## 12. Before committing operational changes

From the repository root:

```bash
git status --short
git diff --check
```

Stage only reviewed source and documentation files. Never stage `.env`, private logs, indexes, or local backups. For architecture and setup details, see `README.md`, `INSTALL.md`, `scripts/README.md`, and `src/retrieval/README.md`.

## Quick reference: serving examples

Run from the repository root with `.venv` active. The example is **development only** and deliberately uses a localhost-only port distinct from production.

```bash
# Terminal on askdunegpvm01: start private development instance.
ENABLE_AUTHENTICATION=false RETRIEVAL_BACKEND=unified PYTHONPATH=. \
  python -m uvicorn main:app --host 127.0.0.1 --port 8001

# Separate terminal on your local machine: forward port 8001.
ssh -N -L 8001:127.0.0.1:8001 askdune

# Check the development endpoint from the server.
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8001/

# Inspect Uvicorn's installed options without starting the server.
python -m uvicorn --help
```

| Setting | Example | Explanation |
|---|---|---|
| `PYTHONPATH=.` | `.` | Include the repository root in Python module lookup |
| `ENABLE_AUTHENTICATION` | `false` | Disable authentication **only for private local development** |
| `RETRIEVAL_BACKEND` | `unified` | Select the unified retrieval implementation |
| `main:app` | `main:app` | Import the FastAPI `app` from `main.py` |
| `--host` | `127.0.0.1` | Accept connections only on the server loopback interface |
| `--port` | `8001` | Development listener; **8000 is reserved for production** |
| `-u` (Python) | `python -u` | Unbuffered console output, useful for long-running scripts |

To stop this foreground development instance, use **Ctrl+C in its own terminal**. Do not run a generic kill command and do not assume the live SSO service uses this startup procedure.
