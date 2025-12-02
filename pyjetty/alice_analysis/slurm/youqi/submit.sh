#! /bin/bash

#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --account=alice
#SBATCH --job-name=youqi
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --array=1-99
#SBATCH --output=/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/slurm-%A_%a.out

FILE_PATHS='/global/cfs/cdirs/alice/youqi/lists/files_LHC18qr.txt'
FILE_PATHS_MC='/global/cfs/cdirs/alice/youqi/lists/files_LHC20g4_568_pthat28.txt'
NFILES=$(wc -l < $FILE_PATHS_MC)
echo "N files to process: ${NFILES}"

FILES_PER_JOB=$(( $NFILES / 99 + 1 ))
echo "Files per job: $FILES_PER_JOB"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

STOP=$(( SLURM_ARRAY_TASK_ID*FILES_PER_JOB ))
START=$(( $STOP - $(( $FILES_PER_JOB - 1 )) ))

if (( $STOP > $NFILES ))
then
  STOP=$NFILES
fi

echo "START = $START"
echo "STOP = $STOP"

for (( JOB_N = $START; JOB_N <= $STOP; JOB_N++ ))
do
  FILE=$(sed -n "$JOB_N"p $FILE_PATHS)
  FILE_MC=$(sed -n "$JOB_N"p $FILE_PATHS_MC)
  srun process_embed_ENC.sh $FILE $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID $JOB_N $FILE_MC
  # srun pythia_gen_embed_ENC.sh $FILE $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID $JOB_N
done

mkdir -p /global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/${SLURM_ARRAY_JOB_ID}/slurm_output 

# Move stdout to appropriate folder
mv /global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out /global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/${SLURM_ARRAY_JOB_ID}/slurm_output 