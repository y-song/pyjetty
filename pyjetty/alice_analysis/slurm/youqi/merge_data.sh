#! /bin/bash
#
# Script to merge output ROOT files

JOB_ID=52300263

FILE_DIR=/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/$JOB_ID
FILES=$( find "$FILE_DIR" -name "*.root" -size +500c)

OUTPUT_DIR=/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/$JOB_ID
hadd $OUTPUT_DIR/AnalysisResultsFinal.root $FILES