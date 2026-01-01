#! /bin/bash

#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --account=alice
#SBATCH --job-name=youqi
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --time=4:30:00
#SBATCH --array=1-499
#SBATCH --output=/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/slurm-%A_%a.out
#SBATCH --mem=60G

INPUT_LIST='/global/cfs/cdirs/alice/youqi/lists/files_LHC18qr.txt'
FILE_PATHS_MC='/global/cfs/cdirs/alice/youqi/lists/files_LHC20g4_568_pthat28.txt'
NFILES=$(wc -l < $FILE_PATHS_MC)
echo "N files to process: ${NFILES}"

FILES_PER_JOB=$(( $NFILES / 499 + 1 ))
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
  # Load modules
  source /global/cfs/cdirs/alice/youqi/pyjetty_env.sh
  
  # Create combined input file by hadding together 200 files
  FILE_COMBINED_LIST=$(shuf -n 200 $INPUT_LIST | tr '\n' ' ')
  FILE_DATA=/pscratch/sd/y/youqi/combined$JOB_N.root
  hadd -f $FILE_DATA $FILE_COMBINED_LIST 2>&1 | grep -v "TBufferFile::ReadObject" | grep -v "TList::Merge"
  # hadd -v 0 $FILE_DATA $FILE_COMBINED_LIST
  
  FILE_MC=$(sed -n "$JOB_N"p $FILE_PATHS_MC)
  srun /global/cfs/cdirs/alice/youqi/mypyjetty/pyjetty/pyjetty/alice_analysis/slurm/youqi/process_embed_ENC.sh $FILE_DATA $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID $JOB_N $FILE_MC
  
  # Remove combined input file
  rm $FILE_DATA
done

mkdir -p /global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/${SLURM_ARRAY_JOB_ID}/slurm_output 

# Move stdout to appropriate folder
mv /global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out /global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/${SLURM_ARRAY_JOB_ID}/slurm_output 