#!/bin/bash
#PBS -l select=8:system=aurora,place=scatter
#PBS -l walltime=3:200:00
#PBS -l filesystems=home
#PBS -A LLM_for_DUNE
#PBS -q prod

export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

#remove export PMI_NO_FORK=1
export BALSAM_SITE_PATH=/lus/flare/projects/LLM_for_DUNE/users/aditya/BALSAM/LLM_DUNEGPT
cd $BALSAM_SITE_PATH

echo "Starting balsam launcher at $(date)"
/lus/flare/projects/LLM_for_DUNE/users/aditya/BALSAM/balsam_venv/bin/python3.10 /lus/flare/projects/LLM_for_DUNE/users/aditya/BALSAM/balsam_venv/lib/python3.10/site-packages/balsam/cmdline/launcher.py -j mpi -t 198  \
 --tag job_start=0  \

echo "Balsam launcher done at $(date)"