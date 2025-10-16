#!/bin/bash
#PBS -l select=1:system=aurora,place=scatter
#PBS -l walltime=1:60:00
#PBS -l filesystems=home
#PBS -A LLM_for_DUNE
#PBS -q debug

export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

#remove export PMI_NO_FORK=1
export BALSAM_SITE_PATH=/lus/flare/projects/LLM_for_DUNE/users/Rishika/LLM_for_DUNE/BALSAM/DUNEGPT_polaris_cursor_balsam
cd $BALSAM_SITE_PATH

echo "Starting balsam launcher at $(date)"
/home/rishikas/python311/bin/python3.11 /home/rishikas/python311/lib/python3.11/site-packages/balsam/cmdline/launcher.py -j mpi -t 58  \
 --tag job_start=1  \

echo "Balsam launcher done at $(date)"