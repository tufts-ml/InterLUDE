#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

export ent_loss_ratio=0.05
export use_class_weights='True'

export num_workers=8
export implementation='InterLUDEPLUS'
export normalization='unnormalized_HWC'

export resume='last_checkpoint.pth.tar'

#experiment setting
export dataset_name='TMED2'
# export resolution=32
export data_seed=2
export development_size='DEV56'
export training_seed=0

export script="YOUR_SCRIPT_PATH"


export arch='wideresnet'
export start_epoch=0
# export num_classes=10


export train_dir="YOUR_EXPERIMENT_DIR"


mkdir -p $train_dir


#data paths
#data paths
export l_train_dataset_path="YOUR_DATA_PATH/train.npy"

export u_train_dataset_path="YOUR_DATA_PATH/u_train.npy"

export val_dataset_path="YOUR_DATA_PATH/val.npy"

export test_dataset_path="YOUR_DATA_PATH/test.npy"



#shared config
export labeledtrain_batchsize=64 #default
export unlabeledtrain_batchsize=320 #default
export em=0 #default


export lr=0.1
export wd=5e-4
export lambda_u_max=0.5
export temperature=1.0
export mu=5

export optimizer_type='SGD'
export lr_schedule_type='CosineLR'
export relativeloss_warmup_schedule_type="NoWarmup"


export pn_strength=0.1
export lambda_relative_loss=0.1


if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch <../do_experiment.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash ../do_experiment.slurm
fi

