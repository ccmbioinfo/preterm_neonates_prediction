#!/bin/bash

for view in axial sagittal coronal
do
  echo ${view}
  qsub -v run='new_resnet18_freeze',view=${view} run_nnet_t1.sh
  #qsub -v run='july_resnet18_freeze',view='sagittal' test.sh
  #qsub -v run='july_resnet18_freeze',view='coronal' test.sh
done
