- Release branch: `release/askdune-v1.0.0`.
- Application directory: `<checkout>/askdune_app`.
- Use an isolated development checkout and database for testing.

# AskDUNE operations runbook

This runbook covers development operations for the AskDUNE unified parent-child retrieval application on `askdunegpvm01` and provides safety guidelines for future production deployment.

It does **not** authorize modifying, stopping, restarting, or replacing the existing production AskDUNE service.

## 1. Hosts, paths, and boundaries

- Development host: `askdunegpvm01.fnal.gov`, account `dune`.
- Release branch: `release/askdune-v1.0.0`.
- Release cleanup worktree: `/exp/dune/app/users/dune/askdune/dev/askdune-v1.0.0/askdune/askdune-v1-cleanup`.
- Application directory in cleanup worktree: `/exp/dune/app/users/dune/askdune/dev/askdune-v1.0.0/askdune/askdune-v1-cleanup/askdune_app`.
- Existing production AskDUNE service uses port **8000** and Fermilab SSO.
- Private development testing uses port **8001**, bound to `127.0.0.1`.

**Never stop, restart, replace, or bind to the existing production port 8000 using this runbook.**

Preserve all existing legacy and experimental ChromaDB indexes. Do not delete, overwrite, or rebuild them without an explicitly reviewed plan.

Never commit `.env`, credentials, ChromaDB data, private logs, or local backup files.

## 2. Prepare a development shell

Connect to the development host and enter the cleanup worktree:

```bash
ssh askdune

cd <checkout>/askdune_app
source .venv/bin/activate
```

If the virtual environment is located elsewhere, activate the appropriate development environment instead.

### Inspect database configuration

The following command prints resolved paths and model names without displaying credentials:

```bash
python - <<'PY'
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

for key in (
    "PROD_DB_PATH",
    "INDEXING_DB_PATH",
    "EMBEDDING_MODEL",
    "RERANKER_MODEL",
    "K_DOCS",
):
    value = os.getenv(key)
    if key.endswith("_PATH") and value:
        print(f"{key}: {Path(value).expanduser().resolve()}")
    else:
        print(f"{key}: {value or '<unset>'}")
PY
```

Confirm that `PROD_DB_PATH` points to a populated unified serving index and that `INDEXING_DB_PATH` points to a different directory.

Do not use the existing legacy production database as the unified serving index.

## 3. Start a private development web server

In a dedicated SSH terminal, from the `askdune_app` directory:

```bash
ENABLE_AUTHENTICATION=false PYTHONPATH=. \
  python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Keep this terminal open.

From a second terminal on your local workstation, establish an SSH tunnel:

```bash
ssh -N -L 8001:127.0.0.1:8001 askdune
```

Open `http://127.0.0.1:8001/` in your local browser.

Do not bind an authentication-disabled development server to `0.0.0.0` or expose it publicly.

### Check the development endpoint

From the development host:

```bash
curl -s -o /dev/null -w '%{http_code}\n' \
  http://127.0.0.1:8001/
```

The application also provides health and statistics endpoints:

```bash
curl -s http://127.0.0.1:8001/api/health
curl -s http://127.0.0.1:8001/api/stats
```

A successful HTTP response does not necessarily mean every dependency is healthy. Inspect the health response body and application logs.

## 4. Stop or inspect the development server

If running in the foreground, press **Ctrl+C in the terminal that started port 8001**.

To inspect the development listener:

```bash
ss -ltnp | grep ':8001'
```

If you no longer control the starting terminal, inspect the exact process and command line before taking action.

Never use broad commands such as `pkill uvicorn` or `killall python`, because they could affect the production service.

## 5. Indexing prerequisites and destination checks

The unified DocDB and Indico indexing scripts write to the configured `INDEXING_DB_PATH` or an explicit `--index-path`.

They do not derive a sibling `parent_child_experiment` directory.

Before starting indexing:

1. Confirm that the necessary DocDB or Indico access credentials are configured.
2. Confirm that the embedding model matches the intended unified index.
3. Resolve and inspect `INDEXING_DB_PATH`, `PROD_DB_PATH`, and any explicit `--index-path`.
4. Verify that the indexing destination is not the existing production database or another important index.
5. Arrange a backup or snapshot when appropriate and ensure that no conflicting writer is running.
6. Begin with a small batch and inspect logs, collection counts, and retrieval behavior before expanding.

The scripts reject an indexing destination whose resolved path matches `PROD_DB_PATH`, but you must still inspect the paths yourself.

### Read-only CLI inspection

```bash
python -m scripts.run_docdb_indexing --help
python -m scripts.run_indico_indexing --help
```

These commands do not index documents.

## 6. Run a small DocDB indexing batch

**Only after completing the checks in Section 5:**

```bash
python -m scripts.run_docdb_indexing \
  --start 1 \
  --limit 5 \
  --batch-size 32
```

The defaults are:

| Option | Default |
| --- | --- |
| `--start` | `1` |
| `--limit` | `5` |
| `--batch-size` | `32` |
| `--index-path` | Uses `INDEXING_DB_PATH` when omitted |

The DocDB pipeline prepares documents and attachments, generates child embeddings, and tracks active document versions using `DUNE_DOCDB_VERSION_MANIFEST`.

A repeated run can help identify duplicate-indexing issues, but it does not by itself validate all document-update and version-replacement behavior.

## 7. Run a small Indico indexing batch

**Only after completing the checks in Section 5:**

```bash
python -m scripts.run_indico_indexing \
  --start 0 \
  --limit 5 \
  --batch-size 32
```

To index one known event:

```bash
python -m scripts.run_indico_indexing \
  --event-id EVENT_ID
```

Replace `EVENT_ID` with a valid event identifier.

The Indico script can skip events whose existing child records are marked `index_status=complete`. Its completion logic differs from the DocDB version-manifest mechanism.

## 8. Background jobs and development logging

For long-running indexing tests, a persistent `tmux` session is preferable to an unmanaged background process.

Check whether `tmux` is installed:

```bash
command -v tmux
```

Start a session:

```bash
tmux new -s askdune-index-test
```

Inside the session, activate the development environment and run the reviewed indexing command.

Detach with **Ctrl+B**, then **D**. Reconnect using:

```bash
tmux attach -t askdune-index-test
```

Use distinct session names for web serving and indexing. Do not reuse production session names.

### Logging

Optionally redirect development indexing output to a private directory:

```bash
mkdir -p "$HOME/askdune-dev-logs"

python -m scripts.run_docdb_indexing \
  --start 1 --limit 5 --batch-size 32 \
  > "$HOME/askdune-dev-logs/docdb-test-$(date +%Y%m%d-%H%M%S).log" 2>&1
```

Check the exit status immediately after completion and inspect the log.

Logs may contain document text or operational details. Do not commit them.

Redirecting output alone does not keep a process alive after logout; use a persistent session when needed.

## 9. Scheduling and incremental indexing

Automatic indexing is not configured by this runbook.

Do not install cron jobs, systemd units, or automated full-corpus indexing based solely on the example commands.

Before scheduling, establish:

- The indexing account, working directory, Python environment, and secret-loading method.
- The exact indexing destination and read/write permissions.
- Incremental discovery, deletion handling, version updates, and rerun behavior.
- A single-writer policy or locking mechanism.
- Retry/backoff, source API rate limits, and failure notifications.
- Logging, disk quotas, backups, and recovery procedures.
- Post-indexing collection and retrieval validation.
- A controlled process for making a validated index available to the serving application.

Scheduled jobs should use a reviewed wrapper with absolute paths and appropriate locking.

## 10. Container deployment

The Dockerfile launches FastAPI through Uvicorn.

Docker Compose uses `HOST_DB_PATH` for the host-side unified index and sets `PROD_DB_PATH=/data/chroma` inside the container.

The configured serving index mount is read-only.

**Container startup and ChromaDB compatibility with the read-only mount have not yet been validated.**

Docker and Podman are unavailable on the current development VM. Container testing must therefore take place in a suitable isolated environment.

Do not point a development container at the existing legacy production database.

## 11. Production service operations

The existing SSO-enabled AskDUNE service is managed separately.

Its production startup mechanism, service name, working directory, restart policy, log locations, and deployment approval process have not been verified for this runbook.

Before documenting production start or stop commands, establish these details with the service owner and record a tested, authorized procedure.

Do not use the unauthenticated development command on the public interface or production port.

## 12. Troubleshooting

| Symptom | First safe check |
| --- | --- |
| Development browser cannot connect | Confirm SSH tunnel, Uvicorn process, and port 8001 listener |
| Application refuses to start | Check `.env`, `PROD_DB_PATH`, expected collections, and Python imports |
| No unified results | Verify index location, embedding compatibility, collection counts, and indexing coverage |
| Indexing refuses to run | Check credentials, embedding model, and the resolved indexing destination |
| Unexpectedly slow search | Inspect model initialization, retrieval logs, and reranking |
| Incorrect collection counts | Stop further test writes and review logs and source-selection logic |
| Port already occupied | Identify the exact listener; do not terminate production processes |
| Container cannot open ChromaDB | Verify mount paths, database permissions, and read-only compatibility |

## 13. Before committing release changes

From the cleanup worktree:

```bash
git status --short
git diff --check
```

Stage only reviewed source files and documentation.

Do not stage `.env`, indexes, test data, private logs, patches, or backup files.

For architecture and configuration details, see `README.md`, `INSTALL.md`, `scripts/README.md`, and `src/retrieval/README.md`.

## Quick reference: development serving

From the `askdune_app` directory:

```bash
# Start private development instance.
ENABLE_AUTHENTICATION=false PYTHONPATH=. \
  python -m uvicorn main:app --host 127.0.0.1 --port 8001

# From a separate terminal on your workstation:
ssh -N -L 8001:127.0.0.1:8001 askdune

# Check the development endpoint from the server:
curl -s -o /dev/null -w '%{http_code}\n' \
  http://127.0.0.1:8001/
```

Stop the foreground development instance using **Ctrl+C in its own terminal**.

Never run a generic process-killing command or assume the live production service uses the same startup procedure.
