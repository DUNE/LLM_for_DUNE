# BALSAM README

## Directory Layout:

BALSAM/LLM_DUNEGPT/
├── balsam-service.pid      # Balsam service process ID
├── data/                   # Stores job output directories
├── job-template.sh         # Optional job script template
├── log/                    # Logs for job submissions and site
├── qsubmit/                # Internal Balsam scheduling metadata
├── scripts/
│   ├── app/                # Registered Balsam application(s)
│   ├── workflow/           # Workflow/job creation logic
│   └── utils/              # Analytics, monitoring tools
├── settings.yml            # Site-level configuration
├── specs/                  # Job specification metadata
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
