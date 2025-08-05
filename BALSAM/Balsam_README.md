# BALSAM README

## Directory Layout:

BALSAM/LLM_DUNEGPT/
├── balsam-service.pid         # Process ID of active Balsam site (if running)
├── settings.yml               # Site-level Balsam configuration
├── job-template.sh            # Optional PBS/SLURM job script template (unused)
├── qsubmit/                   # Internal Balsam job queue metadata
├── log/                       # Logs from site and launcher (stdout, stderr)
├── data/                      # Output directories for job execution
│   └── workdir/               # Per-job output (e.g. logs, checkpoints)
│
├── specs/                     # Metadata about job specs, app registry
│
├── scripts/                   # Custom Balsam logic
│   ├── app/                   # App registration 
│   ├── workflow/              # Job creation/submission scripts
│   │   └── embed_docdb_indico.py     # Creates jobs for batch submission
│   └── utils/                 # Plotting, analytics, monitoring
│       └── analytics.py       # Utilization and throughput plot generator
               # Job specification metadata
Getting Started

## Activate Balsam environment:
source balsam_venv/bin/activate

### FIRST TIME ONLY: Initialize and activate site:
balsam site init LLM_DUNEGPT (names site LLM_DUNEGPT
balsam site activate LLM_DUNEGPT



## Configure job submission
Edit scripts/workflow/embed_docdb_indico.py to define job range, node packing, and duration:
i.e.
begin_index = 0
end_index = 4
spill_size = 4
num_nodes = 4
wall_time_min = 180
Create Balsam jobs:


## Run jobs:
First, register the app: python /scripts/app/embed_docdb_indico.py
Then, make sure your site is active by running balsam site start
Once site is running, submit jobs using python /scripts/workflow/embed_docdb_indico.py
(You can stop the site using balsam site stop)

## Monitor jobs:
balsam queue ls
balsam job ls

## Analytics:
python scripts/utils/analytics.py

if you make edits to the app in /scripts/app/embed_docdb_indico.py, simply stop the site (balsam site stop), rerun: python /scripts/app/embed_docdb_indico.py
and then start the site again and you can submit jobs with the new app setup.
