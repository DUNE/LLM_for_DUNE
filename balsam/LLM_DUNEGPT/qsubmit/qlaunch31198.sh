#!/bin/bash
#PBS -l select=8:system=aurora,place=scatter
#PBS -l walltime=0:50:00
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
/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/bin/python3.10 /home/aditya-singh/.local/aurora/frameworks/2025.0.0/lib/python3.10/site-packages/balsam/cmdline/launcher.py -j mpi -t 48  \
 --tag job_start=0  \

echo "Balsam launcher done at $(date)"