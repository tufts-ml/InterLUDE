#!/usr/bin/env bash

# Pass along all ENV variables as arguments to my Python script
python $script \
    --ent_loss_ratio $ent_loss_ratio\
    --use_class_weights $use_class_weights\
    --num_workers $num_workers\
    --dataset_name $dataset_name\
    --data_seed $data_seed\
    --development_size $development_size\
    --training_seed $training_seed\
    --arch $arch\
    --start_epoch $start_epoch\
    --resume $resume\
    --train_dir $train_dir\
    --l_train_dataset_path $l_train_dataset_path\
    --u_train_dataset_path $u_train_dataset_path\
    --val_dataset_path $val_dataset_path\
    --test_dataset_path $test_dataset_path\
    --labeledtrain_batchsize $labeledtrain_batchsize\
    --unlabeledtrain_batchsize $unlabeledtrain_batchsize\
    --em $em\
    --temperature $temperature\
    --mu $mu\
    --lambda_u_max $lambda_u_max\
    --lambda_relative_loss $lambda_relative_loss\
    --pn_strength $pn_strength\
    --lr $lr\
    --wd $wd\
    --optimizer_type $optimizer_type\
    --lr_schedule_type $lr_schedule_type\
    --relativeloss_warmup_schedule_type $relativeloss_warmup_schedule_type\
    