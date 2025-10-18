#! /bin/bash
#
# Script to merge output ROOT files

JOB_ID=44064427

FILE_DIR=/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/$JOB_ID
FILES=$( find "$FILE_DIR" -name "*.root" )

OUTPUT_DIR=/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/$JOB_ID
hadd -f -j 20 $OUTPUT_DIR/AnalysisResultsFinal.root $FILES
