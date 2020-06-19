#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 9:0
#SBATCH --qos bbshort
#SBATCH --mail-type NONE
#SBATCH --job-name=getdnu
#SBATCH --account=nielsemb-plato-peakbagging
#SBATCH --constraint cascadelake

set -e

module purge; module load bluebear # this line is required
module load matplotlib/3.1.1-fosscuda-2019a-Python-3.7.2

python -u /rds/homes/n/nielsemb/repos/PSM128/getdnu/compute_dnu_from_yu_list.py


