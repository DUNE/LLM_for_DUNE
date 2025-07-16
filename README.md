# LLM_for_DUNE
Large Language Models for DUNE
# Steps to Run a Balsam Job on Aurora

1. SSH into Aurora:
   ssh [username]@aurora.alcf.anl.gov

2. Load Required Modules:
   module load autoconf cmake
   module load frameworks
   module load cmake
   module use /soft/modulefiles

3. Navigate to Project Directory and Activate Environment:
   cd /lus/flare/projects/LLM_for_DUNE/users/aditya/DUNEGPT_polaris
   source new_venv/bin/activate

4. First-Time Environment Setup (only needed once):
   pip3 install -r requirements.txt
   python -m spacy download en_core_web_sm
   source init.sh

5. Register Balsam Application (only needed once):
   cd BALSAM/LLM_DUNEGPT_v2/
   python ./scripts/app/embed_docdb_indico.py

6. Start the Balsam Site (once per session):
   balsam site start

   # Make sure the site is active and named:
   # LLM_DUNEGPT_v2

7. Submit Jobs Using Workflow Script:
   python ./scripts/workflows/embed_docdb_indico.py

   # This creates and submits the Balsam job on Aurora.
