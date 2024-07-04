#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
naming convention:

[labeledtrain, labeledtrain_batchsize, labeledtrain_iter, labeledtrain_loader, labeledtrain_dataset]

[unlabeledtrain, unlabeledtrain_batchsize, unlabeledtrain_iter, unlabeledtrain_loader, unlabeledtrain_dataset]

'''
import argparse
import logging
import math
import os
import random
import shutil
import time
import json


import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter


from libml.randaugment import RandAugmentMC
from libml.utils import save_pickle
from libml.utils import train_one_epoch, eval_model
from libml.models.ema import ModelEMA


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()


parser.add_argument('--num_workers', default=0, type=int)


##########################################################FIXED SETTING##########################################################
parser.add_argument('--use_class_weights', default='True', type=str,
                    help='if use_class_weights is True, set class weights to be tie to combo of development_size and data_seed') 

# parser.add_argument('--train_iterations', default=2**20, type=int, help='total iteration to run, follow previous works')
# parser.add_argument('--iteration_per_epoch', default=1024, type=int)
parser.add_argument('--train_epoch', default=1000, type=int)
parser.add_argument('--lr_warmup_iteration', default=0, type=float,
                    help='warmup iteration for linear rate schedule') #following https://github.com/kekmodel/FixMatch-pytorch/blob/master/train.py
parser.add_argument('--lr_schedule_type', default='CosineLR', choices=['CosineLR', 'FixedLR'], type=str) #FM default cosine


parser.add_argument('--optimizer_type', default='SGD', type=str) 

parser.add_argument('--resume', default='', type=str,
                    help='name of the checkpoint (default: none)') 

parser.add_argument('--resume_checkpoint_fullpath', default='', type=str,
                    help='fullpath of the checkpoint to resume from(default: none)') 

parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')

parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov momentum')


parser.add_argument('--use_ema', action='store_true', default=True,
                    help='use EMA model')

parser.add_argument('--ema_decay', default=0.999, type=float,
                    help='EMA decay rate')

##########################################################Default Hyper##########################################################

#new hyper
parser.add_argument('--lambda_relative_loss', default=1, type=float, help='coefficient of relative loss')

parser.add_argument('--FMLikeSharpening_T', default=1.0, type=float, help='temperature for RelativeCE loss')

parser.add_argument('--relativeloss_warmup_schedule_type', default='NoWarmup', choices=['NoWarmup', 'Linear', 'Sigmoid', ], type=str) 

parser.add_argument('--relativeloss_warmup_pos', default=0.4, type=float, help='position at which relative loss warmup ends') 

#shared config
parser.add_argument('--labeledtrain_batchsize', default=64, type=int)
parser.add_argument('--unlabeledtrain_batchsize', default=448, type=int)
parser.add_argument("--em", default=0, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")

#FM config
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout_rate')

parser.add_argument('--lr', default=0.03, type=float, help='learning rate')

parser.add_argument('--wd', default=5e-4, type=float, help='weight decay') #5e-4 for all CIFAR10 and STL10 experiment, but SimMatchV2 use 1e-3 for CIFAR100

parser.add_argument('--lambda_u_max', default=1, type=float, help='coefficient of unlabeled loss')

parser.add_argument('--temperature', default=0.95, type=float, help='temperature for label guessing')

parser.add_argument('--mu', default=5, type=int,
                    help='coefficient of unlabeled batch size')

parser.add_argument('--threshold', default=0.95, type=float,
                    help='pseudo label threshold')

parser.add_argument('--unlabeledloss_warmup_schedule_type', default='NoWarmup', choices=['NoWarmup', 'Linear', 'Sigmoid', ], type=str)  

parser.add_argument('--unlabeledloss_warmup_pos', default=0.4, type=float, help='position at which unlabeled loss warmup ends') #following MixMatch and FixMatch repo


#####################################################Experiment logistic#########################################################
parser.add_argument('--dataset_name', default='TMED2', type=str, help='name of dataset')

# parser.add_argument('--num_classes', default=10, type=int)
# parser.add_argument('--resolution', default=32, type=int)

parser.add_argument('--data_seed', default=0, type=int, help='random seed data partitioning procedure')
parser.add_argument('--development_size', default='DEV479', help='DEV479, DEV165, DEV56')
parser.add_argument('--training_seed', default=0, type=int, help='random seed for training procedure')
parser.add_argument('--arch', default='wideresnet_scale4', type=str, help='backbone to use')

parser.add_argument('--train_dir', 
                    help='directory to output the result')

parser.add_argument('--l_train_dataset_path', default='', type=str)
parser.add_argument('--u_train_dataset_path', default='', type=str)
parser.add_argument('--val_dataset_path', default='', type=str)
parser.add_argument('--test_dataset_path', default='', type=str)


parser.add_argument('--pn_strength', default=0.1, type=float)


def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise NameError('Bad string')
        

parser.add_argument('--use_DA', type=str2bool, default="False")
parser.add_argument('--clip_norm', default=0.0, type=float)

def prRed(prt): print("\033[91m{}\033[0m" .format(prt))
def prGreen(prt): print("\033[92m{}\033[0m" .format(prt))
def prYellow(prt): print("\033[93m{}\033[0m" .format(prt))
def prLightPurple(prt): print("\033[94m{}\033[0m" .format(prt))
def prPurple(prt): print("\033[95m{}\033[0m" .format(prt))
def prCyan(prt): print("\033[96m{}\033[0m" .format(prt))
def prRedWhite(prt): print("\033[41m{}\033[0m" .format(prt))
def prWhiteBlack(prt): print("\033[7m{}\033[0m" .format(prt))
        

#checked
def save_checkpoint(state, is_best, checkpoint, filename='last_checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))
        
#checked
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
    
#learning rate schedule
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_iterations,
                                    lr_cycle_length, #total train iterations
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_iteration):
        if current_iteration < num_warmup_iterations:
            return float(current_iteration) / float(max(1, num_warmup_iterations))
        no_progress = float(current_iteration - num_warmup_iterations) / \
            float(max(1, float(lr_cycle_length) - num_warmup_iterations))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)    



def get_fixed_lr(optimizer,
                num_warmup_iterations,
                lr_cycle_length, #total train iterations
                num_cycles=7./16.,
                last_epoch=-1):
    def _lr_lambda(current_iteration):
        
        return 1.0

    return LambdaLR(optimizer, _lr_lambda, last_epoch)    
   


def create_model(args):
    
    if args.arch=='wideresnet_scale4':
        import libml.models.wideresnet as models
        model_depth = 28
        model_width = 2
            
        model = models.build_wideresnet(depth=model_depth,
                                        widen_factor=model_width,
                                        dropout=args.dropout_rate,
                                        num_classes=args.num_classes,
                                        pn_strength=args.pn_strength)
        
    else:
        raise NameError('Note implemented yet')
        
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters())/1e6))
    return model



def main(args, brief_summary):
    #define transform for each part of the dataset
    if args.dataset_name == 'TMED2':
        from libml.tmed_data import TMED as dataset
        from libml.tmed_l_u_paired_data import TMED_l_u_paired as paired_dataset

        args.resolution=112
        args.num_classes=4
        
        if args.use_DA:
            raise NameError('SHOULD NOT USE DA')
        else:
            prRed('!!!!!!!!!!!use_DA: {}!!!!!!!!!!!'.format(args.use_DA))
            p_target=None
            p_model=None
            
    else:
        raise NameError('THIS SCRIPT ONLY FOR TMED2')
    
    

    
    transform_labeledtrain = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resolution,
                             padding=int(args.resolution*0.125),
                             padding_mode='reflect'),
        transforms.ToTensor(),
#         transforms.Normalize(mean=means, std=stds)
    ])
    
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
#         transforms.Normalize(mean=means, std=stds)
    ])
    
    
    
    class TransformFixMatch(object):
        def __init__(self, mean=None, std=None):
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=args.resolution,
                                      padding=int(args.resolution*0.125),
                                      padding_mode='reflect')])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=args.resolution,
                                      padding=int(args.resolution*0.125),
                                      padding_mode='reflect'),
                RandAugmentMC(n=2, m=10, resolution=args.resolution)])
            self.normalize = transforms.Compose([
                transforms.ToTensor(),
#                 transforms.Normalize(mean=mean, std=std)
            ])

        def __call__(self, x):
            weak = self.weak(x)
            strong = self.strong(x)
            return self.normalize(weak), self.normalize(strong)

        
        
    
#     l_train_dataset = dataset(args.l_train_dataset_path, transform_fn=transform_labeledtrain)
#     u_train_dataset = dataset(args.u_train_dataset_path, transform_fn=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    paired_l_u_dataset = paired_dataset(args.l_train_dataset_path, args.u_train_dataset_path, u_batchsize_multiplier=args.mu, transform_fn=TransformFixMatch())
    val_dataset = dataset(args.val_dataset_path, transform_fn=transform_eval)
    test_dataset = dataset(args.test_dataset_path, transform_fn=transform_eval)
    

    print('Created dataset')
    print("labeled data : {}, unlabeled data : {}, training data : {}".format(
    paired_l_u_dataset.l_size, paired_l_u_dataset.u_size, len(paired_l_u_dataset)))
    print("val data : {}".format(len(val_dataset)))
    print("test data : {}".format(len(test_dataset)))


    
#     l_loader = DataLoader(l_train_dataset, args.labeledtrain_batchsize, shuffle=True, drop_last=True)
#     u_loader = DataLoader(u_train_dataset, args.unlabeledtrain_batchsize, shuffle=True, drop_last=True)
#     paired_l_u_loader = DataLoader(paired_l_u_dataset, args.labeledtrain_batchsize, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True)
    paired_l_u_loader = DataLoader(paired_l_u_dataset, args.labeledtrain_batchsize, shuffle=True, num_workers=args.num_workers, pin_memory=False)

    val_loader = DataLoader(val_dataset, 128, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=False)

    test_loader = DataLoader(test_dataset, 128, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=False)

    ##########################################################################################################################    
    weights = args.class_weights
    weights = [float(i) for i in weights.split(',')]
    weights = torch.Tensor(weights)
    print('weights used is {}'.format(weights))
    weights = weights.to(args.device)
    
    #create model
    model = create_model(args) #use transform_fn=None, since MM script already applied transform when constructing dataset
    model.to(args.device)
    
    #optimizer_type choice
    if args.optimizer_type == 'SGD':
        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.wd},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                              momentum=0.9, nesterov=args.nesterov)
        
    
    else:
        raise NameError('Not supported optimizer setting')
    
    
    
    #lr_schedule_type choice
    if args.lr_schedule_type == 'CosineLR':
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.lr_warmup_iteration, args.train_iterations)
    
    elif args.lr_schedule_type == 'FixedLR':
        scheduler = get_fixed_lr(optimizer, args.lr_warmup_iteration, args.train_iterations)
    
    else:
        raise NameError('Not supported lr scheduler setting')
        
            
    #instantiate the ema model object
    ema_model = ModelEMA(args, model, args.ema_decay)
    
    args.start_epoch = 0
    
    best_val_ema_Bacc = 0
    best_test_ema_Bacc_at_val = 0
    
    best_val_raw_Bacc = 0
    best_test_raw_Bacc_at_val = 0
    
    #if continued from a checkpoint, overwrite the  
    #                                              start_epoch,
    #                                              model weights, ema model weights
    #                                              optimizer state dict
    #                                              scheduler state dict
    if args.resume_checkpoint_fullpath is not None:
        try:
            os.path.isfile(args.resume_checkpoint_fullpath)
            logger.info("==> Resuming from checkpoint..")
            checkpoint = torch.load(args.resume_checkpoint_fullpath)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])

            best_val_ema_Bacc = checkpoint['best_val_ema_Bacc']
            best_test_ema_Bacc_at_val = checkpoint['best_test_ema_Bacc_at_val']
            
            best_val_raw_Bacc = checkpoint['best_val_raw_Bacc']
            best_test_raw_Bacc_at_val = checkpoint['best_test_raw_Bacc_at_val']
            
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        except:
            print('!!!!Does not have checkpoint yet!!!!')
            
            
    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset_name}")
    logger.info(f"  Num Epochs = {args.train_epoch}")
    logger.info(f"  Batch size per GPU (labeled+unlabeled) = {args.labeledtrain_batchsize + args.unlabeledtrain_batchsize}")
    logger.info(f"  Total optimization steps = {args.train_iterations}")
            
    
    train_loss_dict = dict()
    train_loss_dict['train_total_loss'] = []
    train_loss_dict['labeled_loss'] = []
    train_loss_dict['unlabeled_loss_unscaled'] = []
    train_loss_dict['unlabeled_loss_scaled'] = []
    train_loss_dict['relative_loss_unscaled'] = []
    train_loss_dict['relative_loss_scaled'] = []
    
    
    is_best = False
    global_iteration_count = 0 #used for the LR schedule if using Cosine. And the unlabeled loss warmup schedule if using warmup
    
    
    for epoch in range(args.start_epoch, args.train_epoch):
        val_predictions_save_dict = dict()
        test_predictions_save_dict = dict()
        
        
        #train
        p_model, train_MaxGradientNorm, global_iteration_count, train_total_loss_list, train_labeled_loss_list, train_unlabeled_loss_unscaled_list, train_unlabeled_loss_scaled_list, train_relative_loss_unscaled_list, train_relative_loss_scaled_list = train_one_epoch(args, weights, paired_l_u_loader, model, ema_model, optimizer, scheduler, epoch, global_iteration_count, p_target, p_model)
        
        
        train_loss_dict['train_total_loss'].extend(train_total_loss_list)
        train_loss_dict['labeled_loss'].extend(train_labeled_loss_list)
        train_loss_dict['unlabeled_loss_unscaled'].extend(train_unlabeled_loss_unscaled_list)
        train_loss_dict['unlabeled_loss_scaled'].extend(train_unlabeled_loss_scaled_list)
        train_loss_dict['relative_loss_unscaled'].extend(train_relative_loss_unscaled_list)
        train_loss_dict['relative_loss_scaled'].extend(train_relative_loss_scaled_list)
        
#         save_pickle(os.path.join(args.experiment_dir, 'losses'), 'losses_dict.pkl', train_loss_dict)
        
        #val
        val_loss, val_raw_Bacc, val_ema_Bacc, val_true_labels, val_raw_predictions, val_ema_predictions = eval_model(args, val_loader, model, ema_model.ema, epoch, evaluation_criterion='balanced_accuracy')

        val_predictions_save_dict['raw_Bacc'] = val_raw_Bacc
        val_predictions_save_dict['ema_Bacc'] = val_ema_Bacc
        val_predictions_save_dict['true_labels'] = val_true_labels
        val_predictions_save_dict['raw_predictions'] = val_raw_predictions
        val_predictions_save_dict['ema_predictions'] = val_ema_predictions
#         save_pickle(os.path.join(args.experiment_dir, 'predictions'), 'val_epoch_{}_predictions.pkl'.format(str(epoch)), val_predictions_save_dict)
            
        #test
        test_loss, test_raw_Bacc, test_ema_Bacc, test_true_labels, test_raw_predictions, test_ema_predictions = eval_model(args, test_loader, model, ema_model.ema, epoch, evaluation_criterion='balanced_accuracy')

        test_predictions_save_dict['raw_Bacc'] = test_raw_Bacc
        test_predictions_save_dict['ema_Bacc'] = test_ema_Bacc
        test_predictions_save_dict['true_labels'] = test_true_labels
        test_predictions_save_dict['raw_predictions'] = test_raw_predictions
        test_predictions_save_dict['ema_predictions'] = test_ema_predictions

#         save_pickle(os.path.join(args.experiment_dir, 'predictions'), 'test_epoch_{}_predictions.pkl'.format(str(epoch)), test_predictions_save_dict)

        #record performance at max val balanced accuracy
        if val_raw_Bacc > best_val_raw_Bacc:

            best_val_raw_Bacc = val_raw_Bacc
            best_test_raw_Bacc_at_val = test_raw_Bacc

            save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_raw_val'), 'val_predictions.pkl', val_predictions_save_dict)

            save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_raw_val'), 'test_predictions.pkl', test_predictions_save_dict)
    
        if val_ema_Bacc > best_val_ema_Bacc:
            is_best=True

            best_val_ema_Bacc = val_ema_Bacc
            best_test_ema_Bacc_at_val = test_ema_Bacc

            save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_ema_val'), 'val_predictions.pkl', val_predictions_save_dict)

            save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_ema_val'), 'test_predictions.pkl', test_predictions_save_dict)
        
        
        

        save_checkpoint(
            {
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.ema.state_dict(),
            'best_val_ema_Bacc': best_val_ema_Bacc,
            'best_val_raw_Bacc': best_val_raw_Bacc,
            'best_test_ema_Bacc_at_val': best_test_ema_Bacc_at_val,
            'best_test_raw_Bacc_at_val': best_test_raw_Bacc_at_val,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            }, is_best, args.experiment_dir)

        #return is_best to False
        is_best = False

        logger.info('At RAW Best val , validation/test %.2f %.2f' % (best_val_raw_Bacc, best_test_raw_Bacc_at_val))

        logger.info('At EMA Best val, validation/test %.2f %.2f' % (best_val_ema_Bacc, best_test_ema_Bacc_at_val))
        

        args.writer.add_scalar('train/1.MaxGradientNorm_each_epoch', train_MaxGradientNorm, epoch)

        args.writer.add_scalar('train/2.total_loss', np.mean(train_total_loss_list), epoch)
        args.writer.add_scalar('train/3.labeled_loss', np.mean(train_labeled_loss_list), epoch)
        args.writer.add_scalar('train/4.unlabeled_loss_unscaled', np.mean(train_unlabeled_loss_unscaled_list), epoch)
        args.writer.add_scalar('train/5.unlabele_loss_scaled', np.mean(train_unlabeled_loss_scaled_list), epoch)
        args.writer.add_scalar('train/6.relative_loss_unscaled', np.mean(train_relative_loss_unscaled_list), epoch)
        args.writer.add_scalar('train/7.relative_loss_scaled', np.mean(train_relative_loss_scaled_list), epoch)

        args.writer.add_scalar('val/1.val_raw_Bacc', val_raw_Bacc, epoch)
        args.writer.add_scalar('val/2.val_ema_Bacc', val_ema_Bacc, epoch)
        args.writer.add_scalar('val/3.val_loss', val_loss, epoch)

        args.writer.add_scalar('test/1.test_raw_Bacc', test_raw_Bacc, epoch)
        args.writer.add_scalar('test/2.test_ema_Bacc', test_ema_Bacc, epoch)
        args.writer.add_scalar('test/3.test_loss', test_loss, epoch)

        brief_summary["number_of_data"] = {
        "labeled":paired_l_u_dataset.l_size, "unlabeled":paired_l_u_dataset.u_size, "validation":len(val_dataset),
    "test":len(test_dataset)
    }
        
        brief_summary['best_test_ema_Bacc_at_val'] = best_test_ema_Bacc_at_val 
        brief_summary['best_test_raw_Bacc_at_val'] = best_test_raw_Bacc_at_val
        brief_summary['best_val_ema_Bacc'] = best_val_ema_Bacc
        brief_summary['best_val_raw_Bacc'] = best_val_raw_Bacc

        with open(os.path.join(args.experiment_dir + "brief_summary.json"), "w") as f:
            json.dump(brief_summary, f)

            

    brief_summary["number_of_data"] = {
        "labeled":paired_l_u_dataset.l_size, "unlabeled":paired_l_u_dataset.u_size,"validation":len(val_dataset),
    "test":len(test_dataset)
    }
    
    
    brief_summary['best_test_ema_Bacc_at_val'] = best_test_ema_Bacc_at_val 
    brief_summary['best_test_raw_Bacc_at_val'] = best_test_raw_Bacc_at_val
    brief_summary['best_val_ema_Bacc'] = best_val_ema_Bacc
    brief_summary['best_val_raw_Bacc'] = best_val_raw_Bacc
    
        
    args.writer.close()

    with open(os.path.join(args.experiment_dir + "brief_summary.json"), "w") as f:
        json.dump(brief_summary, f)

if __name__ == '__main__':
    args = parser.parse_args()
    
    cuda = torch.cuda.is_available()
    
    if cuda:
        print('cuda available')
        device = torch.device('cuda')
        args.device = device
        torch.backends.cudnn.benchmark = True
    else:
        raise ValueError('Not Using GPU')
    #     device = "cpu"
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    logger.info(dict(args._get_kwargs()))

    
    if args.training_seed is not None:
        print('setting training seed{}'.format(args.training_seed), flush=True)
        set_seed(args.training_seed)
        
    
    
    ################################################Determining class weights################################################
    
    if args.use_class_weights == 'True':
        print('!!!!!!!!Using pre-calculated class weights!!!!!!!!')
        
        if args.data_seed == 0 and args.development_size == 'DEV56':
            args.class_weights = '0.135,0.356,0.210,0.299'
            
        elif args.data_seed == 1 and args.development_size == 'DEV56':
            args.class_weights = '0.137,0.398,0.192,0.273'
        
        elif args.data_seed == 2 and args.development_size == 'DEV56':
            args.class_weights = '0.134,0.386,0.203,0.277'
        
        else:
            raise NameError('not valid class weights setting')
    
    else:
        args.class_weights = '0.25,0.25,0.25,0.25'
        print('?????????Not using pre-calculated class weights?????????')

    ################################################Determining nimg_per_epoch################################################
    
    if args.data_seed == 0 and args.development_size == 'DEV56':
        args.nimg_per_epoch = 632 + 239 + 405 + 285
    
    elif args.data_seed == 1 and args.development_size == 'DEV56':
        args.nimg_per_epoch = 650 + 223 + 462 + 325

    elif args.data_seed == 2 and args.development_size == 'DEV56':
        args.nimg_per_epoch = 572 + 199 + 379 + 277
    
    else:
        raise NameError('What is nimg_per_epoch?')

    
     #calculate train_iteration from the designed train_epoch,labeled samples per epoch and labeled batch size
    #calculate iteration_per_epoch using labeled samples per epoch / labeled batch size
    
    args.train_iterations = int(args.train_epoch*args.nimg_per_epoch//args.labeledtrain_batchsize)
    
    args.iteration_per_epoch = int(args.nimg_per_epoch//args.labeledtrain_batchsize)
    print('designated train iterations: {}, iteration_per_epoch: {}'.format(args.train_iterations, args.iteration_per_epoch))
    
    
    experiment_name = "LambdaRelativeLoss-{}_RlossWarmupType-{}_PN-{}_LambdaUMax-{}_lr-{}_wd-{}".format(args.lambda_relative_loss, args.relativeloss_warmup_schedule_type, args.pn_strength, args.lambda_u_max, args.lr, args.wd)

    
    args.experiment_dir = os.path.join(args.train_dir, experiment_name)
    
    if args.resume != 'None':
        args.resume_checkpoint_fullpath = os.path.join(args.experiment_dir, args.resume)
        print('args.resume_checkpoint_fullpath: {}'.format(args.resume_checkpoint_fullpath))
    else:
        args.resume_checkpoint_fullpath = None
        
    os.makedirs(args.experiment_dir, exist_ok=True)
    args.writer = SummaryWriter(args.experiment_dir)
    
    #brief summary:
    brief_summary = {}
    brief_summary['dataset_name'] = args.dataset_name
    brief_summary['algorithm'] = 'InterLUDE'
    brief_summary['hyperparameters'] = {
        'RlossWarmupType':args.relativeloss_warmup_schedule_type,
        'lambda_relative_loss':args.lambda_relative_loss,
        'PN':args.pn_strength,
        'lambda_u_max': args.lambda_u_max,        
        'lr': args.lr,
        'wd': args.wd,
        
    }

    main(args, brief_summary)
    
    
    
    


    
