#! /bin/bash

#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --account=alice
#SBATCH --job-name=youqi
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=5
#SBATCH --time=10:00:00
#SBATCH --array=1-99
#SBATCH --output=/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/slurm-%A_%a.out
#SBATCH --mem=50G

FILE_PATH_DATA='/global/cfs/cdirs/alice/youqi/lists/files_LHC18qr_randomly_hadd50.txt' #'/global/cfs/cdirs/alice/youqi/lists/files_LHC18qr.txt'
FILE_PATH_MC='/global/cfs/cdirs/alice/youqi/lists/files_LHC20g4_568_pthat28.txt'
PROCESS_SCRIPT='/global/cfs/cdirs/alice/youqi/mypyjetty/pyjetty/pyjetty/alice_analysis/process/user/youqi/embed_cs_subtraction.py'

echo "Batch script: submit_embed.sh"
echo "Process script: $PROCESS_SCRIPT"
echo "--------------------------------------------------------------------------
Job info"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
NFILES=$(wc -l < $FILE_PATH_MC)
FILES_PER_TASK=$(( $NFILES / 99 + 1 ))
echo "N MC files: ${NFILES}, Files per task: $FILES_PER_TASK"
echo "--------------------------------------------------------------------------
Task info"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
STOP=$(( SLURM_ARRAY_TASK_ID*FILES_PER_TASK ))
START=$(( $STOP - $(( $FILES_PER_TASK - 1 )) ))
if (( $STOP > $NFILES ))
then
  STOP=$NFILES
fi
echo "Starting at file ID: $START"
echo "Stopping at file ID: $STOP"

# Load modules
source /global/cfs/cdirs/alice/youqi/pyjetty_env.sh

for (( FILE_ID = $START; FILE_ID <= $STOP; FILE_ID++ ))
do
  FILE_DATA=$(sed -n "$FILE_ID"p $FILE_PATH_DATA)
  FILE_MC=$(sed -n "$FILE_ID"p $FILE_PATH_MC)
  echo "--------------------------------------------------------------------------
File info"
  srun /global/cfs/cdirs/alice/youqi/mypyjetty/pyjetty/pyjetty/alice_analysis/slurm/youqi/process_embed_ENC.sh $FILE_DATA $SLURM_ARRAY_JOB_ID $FILE_ID $FILE_MC $PROCESS_SCRIPT
done

# Move stdout to appropriate folder
mkdir -p /global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/${SLURM_ARRAY_JOB_ID}/slurm_output 
mv /global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out /global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/${SLURM_ARRAY_JOB_ID}/slurm_output 