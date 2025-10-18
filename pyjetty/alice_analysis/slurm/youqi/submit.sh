#! /bin/bash

#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --account=alice
#SBATCH --job-name=youqi_test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --array=1-100
#SBATCH --output=/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/slurm-%A_%a.out

FILE_PATHS='/global/cfs/cdirs/alice/youqi/files_LHC18qr_1000.txt'
NFILES=$(wc -l < $FILE_PATHS)
echo "N files to process: ${NFILES}"

FILES_PER_JOB=$(( $NFILES / 10 + 1 ))
echo "Files per job: $FILES_PER_JOB"

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
  srun pythia_gen_ENC_mb.sh $FILE $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID $JOB_N
done

mkdir -p /global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/${SLURM_ARRAY_JOB_ID}/slurm_output 

# Move stdout to appropriate folder
mv /global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out /global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/${SLURM_ARRAY_JOB_ID}/slurm_output 