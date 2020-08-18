## Pre-term Neonates - using T1 (and T2s) to predict adverse outcomes 
 Delvin So
 
### Set-up
1. Git clone this environment, preferably somewhere like your `home` directory on the hpf
2. Install conda environment using `conda env create -f environment.yml`
3. Preprocess images using `prep_v0*_stretch.sh`
    * the input files can be found in `./data/ubc_misc/ubc_masks_v0*, ./data/ubc_misc/ubc_ss_v0*, and ./data/ubc_misc/Prem-UBC_WMI`
    * output will be dumped into `./output/`
4. Neural nets can be run using `run_nnet.sh` (multitask MRNET; see below)

There should be a few absolute directories in `prep_v0*_stretch.sh and run_nnet.sh and` which need to be changed to reflect your own set-up.

### Code Description

- `analysis/prep_v01_stretch.sh` and `analysis/prep_v02_stretch.sh` will run the following two scripts for image pre-processing
    - prepro_pipeline2.py
        - given directories to images, TCV/skull strip masks, white matter injury segmentation masks, (all under the same parent directory!) outputs .npy files and .tifs ready for outcome prediction and white matter injury segmentation, respectively
        - pseudo-code, for each volume:
            - re-windows to level 400, width 200
            - for each slice in the volume, for each of the 3 planes (X, Y, Z),
                - crops the brain, centers, pads each slice to 128x128, trimming start and end slices of the volume using mean pixel intensity as a threshold
            - applies the same operation to a corresponding mask
        - outputs several images and diagnostics for each step of the pre-processing which is dumped into `./figures`, eg. 
    - dump_npy_to_tif.py
        - dumps the numpy images to tif for white matter injury segmentation model

- `analysis/net/*`, contains all neural net related code. the current neural net can be multitask or a single-task and is specified using the `--outcome` argument
    - `net.py`
    - `train_and_eval.py` - contains all the code necessary to instantiate data loaders, models, etc. and run the neural net, passing all arguments
        - possible TODO: take in a json file as a config file (sorry I wrote these scripts before I knew that could be done. It is fairly easy to implement)
    - `run_model.py`
    - `helpers.py` 
    - `data_loader.py` - data loader for the `.npy` arrays. converts each array of `(s x W x H)` to `(s x 3 x W x H)`


## To run a neural net (multitask MRNET)
`qsub run_nnet.sh` OR `./submit_nnet.sh` for all 3 views
```
- all of the below assumes the pre-procssed .npy arrays are stored as 
${root_path}/{some_dir_name1/{some_dir_name2} # unfortunately this is hardcoded in `data_loader.py`
|-- axial
    |-- birth_BC0001_ss.npy
    |-- birth_BC0003_ss.npy
    |-- birth_BC0004_ss.npy
    ....
    |-- birth_BC000N_ss.npy
|-- coronal
    |-- birth_BC0001_ss.npy
    |-- birth_BC0003_ss.npy
    |-- birth_BC0004_ss.npy
    ....
    |-- birth_BC000N_ss.npy
|-- sagittal
    |-- birth_BC0001_ss.npy
    |-- birth_BC0003_ss.npy
    |-- birth_BC0004_ss.npy
    ....
    |-- birth_BC000N_ss.npy
```

- outcome can be multitask or single task by specifiying one of the many outcomes in the spreadsheet found in `manifest_path`
- if outcome = 'multitask', set none to multitask. 
- view - one of 'axial', 'coronal', or 'sagittal'.
- metrics_every_iter - for every i batches, print to screen the current batch loss and metric

```
python3 /home/delvinso/neuro/analysis/net/train_and_eval.py \
     --root_path='/home/delvinso/neuro' \
     --outcome='multitask' \
     --view=${view}  \
     --num_epochs=100 \
     --manifest_path='/home/delvinso/neuro/output/ubc_npy_outcomes_v3_ss_PD.csv' \
     --model_out='/home/delvinso/neuro/output/models'   \
     --metrics_every_iter=20 \
     --task=multitask \
     --run_name="${run_name}"
```
### Output
Output will be stored as `--model_out/${outcome}-${view}/{$run_name}` with the following files:
```
-rw-r--r-- 1 delvinso    16219 Aug  4 11:47 best_epoch_training_results.csv
-rw-r--r-- 1 delvinso     3239 Aug  4 11:47 best_epoch_validation_results.csv
-rw-r--r-- 1 delvinso 44813035 Aug  4 11:47 best.pth.tar
-rw-r--r-- 1 delvinso 44813037 Aug  4 12:10 last.pth.tar
-rw-r--r-- 1 delvinso   268254 Aug  4 12:10 train.log
```

### Important Things
- use the `-q gpu` flag to work with GPUs on the hpf, eg. see below
- `qstat | grep gpu` OR `qstat -q gpu` # see GPU availability
- `qsub -q gpu -l nodes=1:ppn=1:gpus=1,vmem=120g,mem=80g,walltime=48:00:00 -I` # gpu in interactive mode, great for debugging
- sanity check your dataloader/batches going into your neural nets!!

### What needs to be done
- The codebase for pre-processing and the program/scripts for running the neural net is largely finished. Future work requires (hopefully) slight tweaks to the logic to test different architectures.
- Apply pre-processing to T2s as outlined below and in `prepro_pipeline2.py`. These are 2D whereas the T1s are 3d. Most of these are axial so the logic will need to be re-structured.
    - It may also just work out of the box.. but the other views will need to be discarded.
- Voxel based neural net?
- It may also help to check with Lauren to see what else needs to be done.


