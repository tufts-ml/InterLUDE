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

export version='class_dependent'
export num_workers=8
export implementation='InterLUDE'

export resume='last_checkpoint.pth.tar'

#experiment setting
export dataset_name='cifar10'
export nlabels=250
# export resolution=32
export data_seed=0
export training_seed=0

export script="YOUR_SCRIPT_PATH"


export arch='wideresnet'
export start_epoch=0
# export num_classes=10


export train_dir="YOUR_EXPERIMENT_DIR"

mkdir -p $train_dir


#data paths
export l_train_dataset_path="YOUR_DATA_PATH/l_train.npy"

export u_train_dataset_path="YOUR_DATA_PATH/u_train.npy"

export test_dataset_path="YOUR_DATA_PATH/test.npy"


#shared config
export labeledtrain_batchsize=64 #default
export unlabeledtrain_batchsize=448 #default
export em=0 #default


export lr=0.03
export wd=5e-4
export lambda_u_max=1.0
export temperature=1.0
export mu=7
export threshold=0.95

export optimizer_type='SGD'
export lr_schedule_type='CosineLR'
export relativeloss_warmup_schedule_type="NoWarmup"

export pn_strength=0.1
export lambda_relative_loss=1.0




if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch <../do_experiment.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash ../do_experiment.slurm
fi

