#! /bin/bash

# This script takes an input file path as an argument, and runs a python script to
# process the input file and write an output ROOT file.
# The main use is to give this script to a slurm script.

if [ "$1" != "" ]; then
  INPUT_DATA_FILE=$1
  echo "Input data file: $INPUT_DATA_FILE"
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
  FILE_ID=$3
  echo "File ID: $FILE_ID"
else
  echo "Wrong command line arguments"
fi

if [ "$4" != "" ]; then
  INPUT_MC_FILE=$4
  echo "Input MC file: $INPUT_MC_FILE"
else
  echo "Wrong command line arguments"
fi

if [ "$5" != "" ]; then
  PROCESS_SCRIPT=$5
  echo "Process script: $PROCESS_SCRIPT"
else
  echo "Wrong command line arguments"
fi

# Define output path from relevant sub-path of input file
OUTPUT_PREFIX="AnalysisResults/youqi/$JOB_ID"
# Note: suffix depends on file structure of input file -- need to edit appropriately for each dataset
# NUMBER="${INPUT_FILE#*combined}"
# OUTPUT_SUFFIX="${NUMBER%.root}"
# OUTPUT_SUFFIX=$(echo $INPUT_FILE | cut -d/ -f11-15)
OUTPUT_DIR="/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/$OUTPUT_PREFIX/$FILE_ID"
echo "Output dir: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

# Run python script via pipenv
python $PROCESS_SCRIPT -c /global/cfs/cdirs/alice/youqi/mypyjetty/pyjetty/pyjetty/alice_analysis/config/ENC/PbPb/process_data.yaml -f $INPUT_DATA_FILE -o $OUTPUT_DIR -fmc $INPUT_MC_FILE