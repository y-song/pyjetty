#! /bin/bash

# This script takes an input file path as an argument, and runs a python script to
# process the input file and write an output ROOT file.
# The main use is to give this script to a slurm script.

# Take two command line arguments -- (1) input file path, (2) output dir prefix
if [ "$1" != "" ]; then
  INPUT_FILE=$1
  echo "Input file: $INPUT_FILE"
else
  echo "Wrong command line arguments"
fi

if [ "$2" != "" ]; then
  JOB_ID=$2
  echo "Job ID: $JOB_ID"
else
  echo "Wrong command line arguments"
fi

if [ "$3" != "" ]; then
  TASK_ID=$3
  echo "Task ID: $TASK_ID"
else
  echo "Wrong command line arguments"
fi

if [ "$4" != "" ]; then
  FILE_N=$4
  SEED=$(( $FILE_N + 0 ))
  echo "Seed: $SEED"
else
  echo "Wrong command line arguments"
fi

if [ "$5" != "" ]; then
  INPUT_FILE_MC=$5
  echo "Input file MC: $INPUT_FILE_MC"
else
  echo "Wrong command line arguments"
fi

# Define output path from relevant sub-path of input file
OUTPUT_PREFIX="AnalysisResults/youqi/$JOB_ID"
# Note: suffix depends on file structure of input file -- need to edit appropriately for each dataset
OUTPUT_SUFFIX=$(echo $INPUT_FILE | cut -d/ -f11-15)
OUTPUT_DIR="/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/$OUTPUT_PREFIX/$OUTPUT_SUFFIX"
echo "Output dir: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

# Load modules
BASR_DIR="/global/cfs/cdirs/alice/youqi/mypyjetty"
cd $BASR_DIR
source /global/cfs/cdirs/alice/youqi/pyjetty_env.sh

# Run python script via pipenv
cd ${BASR_DIR}/pyjetty/pyjetty/alice_analysis
python /global/cfs/cdirs/alice/youqi/mypyjetty/pyjetty/pyjetty/alice_analysis/process/user/youqi/process_embed_ENC.py -c /global/cfs/cdirs/alice/youqi/mypyjetty/pyjetty/pyjetty/alice_analysis/config/ENC/PbPb/process_data.yaml -f $INPUT_FILE -o $OUTPUT_DIR -fmc $INPUT_FILE_MC