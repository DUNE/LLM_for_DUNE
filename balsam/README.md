# BALSAM Workflow for LLM_DUNEGPT

This directory defines a high-throughput Balsam job workflow for executing document embedding and indexing pipelines (e.g., from DUNE DocDB and Indico) using the LLM_DUNEGPT framework on HPC systems like Aurora or Polaris.

It enables parallelized job scheduling, monitoring, and analytics via the Balsam service.

---

## Directory Structure

```
balsam/
├── job-template.sh         # PBS/SLURM-compatible script template (optional)
├── settings.yml            # Site configuration (site name, database, etc.)
├── balsam-service.pid      # Process ID of running balsam site (auto-generated)
├── data/                   # Output directories from each job
│   └── workdir/            # Logs, stdout, stderr for individual job executions
├── log/                    # Site & launcher logs (stdout, stderr)
├── qsubmit/                # Internal Balsam job queue metadata (auto-managed)
├── specs/                  # App registration and job spec definitions
├── scripts/
│   ├── app/                # App definition script (e.g., FAISS indexer)
│   │   └── embed_docdb_indico.py
│   ├── workflow/           # Job creation and submission logic
│   │   └── embed_docdb_indico.py
│   └── utils/              # Analytics and helper scripts
│       └── analytics.py
```

---

## Quick Start

### Prerequisites

- Python environment with Balsam installed (e.g., `balsam_env`)
- A valid `embed_docdb_indico.py` script that creates and runs FAISS embedding jobs

---

##  Setup

### 1. Activate the Python environment (from /balsam folder)

```bash
source balsam_venv/bin/activate
```

### 2. Initialize the Balsam site (first time only), creates LLM_DUNEGPT folder

```bash
balsam site init LLM_DUNEGPT
```

### 3. Activate the site

```bash
balsam site activate LLM_DUNEGPT
```

---

## Configuration

Edit the job creation script at:

```
scripts/workflow/embed_docdb_indico.py
```

Set these variables to control job parameters:

```python
begin_index = 0         # Start document index
end_index = 4           # End document index (exclusive)
spill_size = 4          # Number of documents per job
num_nodes = 4           # Nodes per job
wall_time_min = 180     # Wall time in minutes
```

---

## Register App

Before submitting jobs, register the app once (from /balsam/LLM_DUNEGPT/):

```bash
python scripts/app/embed_docdb_indico.py
```

---

## Run Jobs

### 1. Start the Balsam service

```bash
balsam site start
```

### 2. Submit jobs

```bash
python scripts/workflow/embed_docdb_indico.py
```

### 3. Monitor job queue

```bash
balsam queue ls
balsam job ls
```

### 4. Stop the service when finished

```bash
balsam site stop
```

---

## Analytics

After jobs complete, generate a utilization plot:

```bash
python scripts/utils/analytics.py
```

This will show node usage and throughput across the duration of the workflow.

---

## Updating App Definition

If you edit `scripts/app/embed_docdb_indico.py`:

1. Stop the site:

```bash
balsam site stop
```

2. Reregister the app:

```bash
python scripts/app/embed_docdb_indico.py
```

3. Restart the site:

```bash
balsam site start
```

4. Submit new jobs as needed.

---

##  Troubleshooting

| Issue                         | Solution |
|------------------------------|----------|
| No jobs appear in queue      | Ensure `balsam site start` was run before submitting |
| Jobs fail silently           | Check logs in `data/workdir/` and `log/` folders |
| Analytics script fails       | Make sure job data has been written to `data/` |
| Site won’t start             | Check for stale `balsam-service.pid` and delete it manually if needed |

---

## Notes

- `settings.yml` is created automatically during site init and should be committed for reproducibility
- You can modify `job-template.sh` to add custom job prolog/epilog (e.g., `module load`, scratch setup)
- All jobs will write logs to `data/workdir/` and internal states to `qsubmit/`

---

## Example Workflow

```bash
# Setup
source balsam_venv/bin/activate
balsam site init LLM_DUNEGPT
balsam site activate LLM_DUNEGPT

# Register app
python scripts/app/embed_docdb_indico.py

# Start Balsam service
balsam site start

# Submit jobs
python scripts/workflow/embed_docdb_indico.py

# Monitor
balsam queue ls
balsam job ls

# Analyze
python scripts/utils/analytics.py

# Cleanup
balsam site stop
```






---

**BALSAM + LLM_DUNEGPT**: Scalable, reliable, and fast document processing for the DUNE collaboration.
