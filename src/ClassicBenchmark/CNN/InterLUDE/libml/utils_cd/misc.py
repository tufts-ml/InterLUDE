import time
from tqdm import tqdm
import torch.nn.functional as F

import logging
import numpy as np
import os
import pickle

import torch

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'AverageMeter', 'train_one_epoch', 'eval_model', 'save_pickle', 'calculate_plain_accuracy']

#https://github.com/google-research/fixmatch/issues/20
def interleave(l_weak, l_strong, u_weak, u_strong, takeN_l_weak, takeN_l_strong, takeN_u_weak, takeN_u_strong):
    
    l_weak_size = len(l_weak)
    l_strong_size = len(l_strong)
    u_weak_size = len(u_weak)
    u_strong_size = len(u_strong)
    
    total_size = l_weak_size + l_strong_size + u_weak_size + u_strong_size
    
    global_index_l_weak = []
    global_index_l_strong = []
    global_index_u_weak = []
    global_index_u_strong = []
    
    global_index = 0
    
    new_x = []
    l_weak_pointer=0
    l_strong_pointer=0
    u_weak_pointer=0
    u_strong_pointer=0
  
    
    while len(new_x)<total_size:
        ############################################################
        #takeN_l_weak
        if l_weak_pointer + takeN_l_weak <= l_weak_size:
            
            new_x.append(l_weak[l_weak_pointer:l_weak_pointer+takeN_l_weak])
            l_weak_pointer += takeN_l_weak
            global_index_l_weak.extend(list(np.arange(global_index, global_index+takeN_l_weak)))
            global_index += takeN_l_weak
            
        else: 
            lastN_l_weak = l_weak_size - l_weak_pointer
            new_x.append(l_weak[l_weak_pointer:l_weak_pointer+lastN_l_weak])
            l_weak_pointer += lastN_l_weak
            global_index_l_weak.extend(list(np.arange(global_index, global_index+lastN_l_weak)))
            global_index += lastN_l_weak
            
        ############################################################
        #takeN_u_weak
        if u_weak_pointer + takeN_u_weak <= u_weak_size:
            
            new_x.append(u_weak[u_weak_pointer:u_weak_pointer+takeN_u_weak])
            u_weak_pointer += takeN_u_weak
            global_index_u_weak.extend(list(np.arange(global_index, global_index+takeN_u_weak)))
            global_index += takeN_u_weak
            
        else: 
            lastN_u_weak = u_weak_size - u_weak_pointer
            new_x.append(u_weak[u_weak_pointer:u_weak_pointer+lastN_u_weak])
            u_weak_pointer += lastN_u_weak
            global_index_u_weak.extend(list(np.arange(global_index, global_index+lastN_u_weak)))
            global_index += lastN_u_weak
        
        
        ############################################################
        #takeN_l_strong
        if l_strong_pointer + takeN_l_strong <= l_strong_size:
            
            new_x.append(l_strong[l_strong_pointer:l_strong_pointer+takeN_l_strong])
            l_strong_pointer += takeN_l_strong
            global_index_l_strong.extend(list(np.arange(global_index, global_index+takeN_l_strong)))
            global_index += takeN_l_strong
            
        else: 
            lastN_l_strong = l_strong_size - l_strong_pointer
            new_x.append(l_strong[l_strong_pointer:l_strong_pointer+lastN_l_strong])
            l_strong_pointer += lastN_l_strong
            global_index_l_strong.extend(list(np.arange(global_index, global_index+lastN_l_strong)))
            global_index += lastN_l_strong
            
            
        ############################################################
        #takeN_u_strong
        if u_strong_pointer + takeN_u_strong <= u_strong_size:
            
            new_x.append(u_strong[u_strong_pointer:u_strong_pointer+takeN_u_strong])
            u_strong_pointer += takeN_u_strong
            global_index_u_strong.extend(list(np.arange(global_index, global_index+takeN_u_strong)))
            global_index += takeN_u_strong
            
        else: 
            lastN_u_strong = u_strong_size - u_strong_pointer
            new_x.append(u_strong[u_strong_pointer:u_strong_pointer+lastN_u_strong])
            u_strong_pointer += lastN_u_strong
            global_index_u_strong.extend(list(np.arange(global_index, global_index+lastN_u_strong)))
            global_index += lastN_u_strong
            
    
    new_x = torch.concat(new_x)
    print(new_x.shape)
    assert len(new_x) == total_size
    
    return new_x, global_index_l_weak, global_index_l_strong, global_index_u_weak, global_index_u_strong
        
        
            
# def interleave(x, size):
#     s = list(x.shape)
#     print('Inside interleave, s is {}, size is {}'.format(s, size))#Inside interleave, s is [1024, 3, 32, 32], size is 16
    
#     return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


# def de_interleave(x, size):
#     s = list(x.shape)
#     print('Inside de_interleave, s is {}, size is {}'.format(s, size)) #s is [1024, 10], size is 16
    
#     return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def prRed(prt): print("\033[91m{}\033[0m" .format(prt))
def prGreen(prt): print("\033[92m{}\033[0m" .format(prt))
def prYellow(prt): print("\033[93m{}\033[0m" .format(prt))
def prLightPurple(prt): print("\033[94m{}\033[0m" .format(prt))
def prPurple(prt): print("\033[95m{}\033[0m" .format(prt))
def prCyan(prt): print("\033[96m{}\033[0m" .format(prt))
def prRedWhite(prt): print("\033[41m{}\033[0m" .format(prt))
def prWhiteBlack(prt): print("\033[7m{}\033[0m" .format(prt))
     
def train_one_epoch(args, paired_l_u_loader, model, ema_model, optimizer, scheduler, epoch, global_iteration_count, p_target, p_model):
    
    MeanGradientNorm_this_epoch, TotalLoss_this_epoch, LabeledLoss_this_epoch, UnlabeledLossUnscaled_this_epoch, UnlabeledLossScaled_this_epoch, RelativeLossUnscaled_this_epoch, RelativeLossScaled_this_epoch, = [], [], [], [], [], [], []
    
    end_time = time.time()
    
    paired_l_u_iter = iter(paired_l_u_loader)
    
    model.train()
    
    MeanGradientNorm = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss = AverageMeter()
    labeled_loss = AverageMeter()
    unlabeled_loss_unscaled = AverageMeter()
    unlabeled_loss_scaled = AverageMeter()
    relative_loss_unscaled = AverageMeter()
    relative_loss_scaled = AverageMeter()
    mask_probs = AverageMeter() #how frequently the unlabeled samples' confidence score greater than pre-defined threshold
    
    
    p_bar = tqdm(range(args.iteration_per_epoch), disable=False)
    
    
    for batch_idx in range(args.iteration_per_epoch):
        
        global_iteration_count+=1
        args.writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_iteration_count)
        
        #unlabeledloss warmup schedule choice
        if args.unlabeledloss_warmup_schedule_type == 'NoWarmup':
            current_warmup = 1
        elif args.unlabeledloss_warmup_schedule_type == 'Linear':
            current_warmup = np.clip(global_iteration_count/(float(args.unlabeledloss_warmup_pos) * args.train_iterations), 0, 1)
        elif args.unlabeledloss_warmup_schedule_type == 'Sigmoid':
            current_warmup = math.exp(-5 * (1 - min(global_iteration_count/(float(args.unlabeledloss_warmup_pos) * args.train_iterations), 1))**2)
        else:
            raise NameError('Not supported unlabeledloss warmup schedule')
            
            
        #unlabeledloss warmup schedule choice
        if args.relativeloss_warmup_schedule_type == 'NoWarmup':
            current_relative_loss_warmup = 1
        elif args.relativeloss_warmup_schedule_type == 'Linear':
            current_relative_loss_warmup = np.clip(global_iteration_count/(float(args.relativeloss_warmup_pos) * args.train_iterations), 0, 1)
        elif args.relativeloss_warmup_schedule_type == 'Sigmoid':
            current_relative_loss_warmup = math.exp(-5 * (1 - min(global_iteration_count/(float(args.relativeloss_warmup_pos) * args.train_iterations), 1))**2)
        else:
            raise NameError('Not supported unlabeledloss warmup schedule')
            
            
        
        
        try:
            (l_weak, l_strong, l_labels), (u_weaks, u_strongs) = paired_l_u_iter.next()
        except:
            paired_l_u_iter = iter(paired_l_u_loader)
            (l_weak, l_strong, l_labels), (u_weaks, u_strongs) = paired_l_u_iter.next()
            
            
#         print('l_weak shape: {}'.format(l_weak.shape)) #--torch.Size([64, 3, 32, 32])
#         print('l_strong shape: {}'.format(l_strong.shape)) #--torch.Size([64, 3, 32, 32])
#         print('l_label shape: {}'.format(l_label.shape)) #--torch.Size([64])
#         print('u_weaks shape: {}'.format(u_weaks.shape)) #--torch.Size([64, 7, 3, 32, 32])
#         print('u_strongs shape: {}'.format(u_strongs.shape)) #--torch.Size([64, 7, 3, 32, 32])

        
        data_time.update(time.time() - end_time)
        
        ##############################################################################################################
        #For FM:
        #reference: https://github.com/kekmodel/FixMatch-pytorch/blob/master/train.py
        

        takeN_l_weak, takeN_l_strong, takeN_u_weak, takeN_u_strong = 1, 1, args.mu, args.mu
    
    
        inputs, global_index_l_weak, global_index_l_strong, global_index_u_weak, global_index_u_strong = interleave(l_weak, l_strong, u_weaks.reshape(-1,3,args.resolution,args.resolution), u_strongs.reshape(-1,3,args.resolution,args.resolution), takeN_l_weak, takeN_l_strong, takeN_u_weak, takeN_u_strong)
        
        inputs = inputs.to(args.device)
        l_labels = l_labels.to(args.device).long()
        
        logits = model(inputs)
        
        logits_l_weak = logits[global_index_l_weak]
        logits_l_strong = logits[global_index_l_strong]
        logits_u_weaks = logits[global_index_u_weak]
        logits_u_strongs = logits[global_index_u_strong]
#         print('logits_l_weak: {}, logits_l_strong: {}, logits_u_weaks: {}, logits_u_strongs: {}'.format(logits_l_weak.shape, logits_l_strong.shape, logits_u_weaks.shape, logits_u_strongs.shape))
        

            
        assert len(logits_l_weak) == len(logits_l_strong)
        assert len(logits_u_weaks) == len(logits_u_strongs)
        
        del logits
        
        labeledtrain_loss = F.cross_entropy(logits_l_weak, l_labels, reduction='mean')
        
        #label guessing
        pseudo_label = torch.softmax(logits_u_weaks.detach()/args.temperature, dim=-1)
        
        if args.use_DA: 
            prGreen('!!!!!!!!!!!use_DA: {}!!!!!!!!!!!'.format(args.use_DA))
            if p_model==None:
                p_model=torch.mean(pseudo_label.detach(), dim=0)
#                 prCyan('First iteration, p_model: {}'.format(p_model))
            else:
                p_model = p_model * 0.999 + torch.mean(pseudo_label.detach(), dim=0) * 0.001
            
            #if use DA, then use p_model and p_target to scale the pseudo_label
#             prCyan('Before scaling with DA, pseudo_label: {} shape: {}'.format(pseudo_label, pseudo_label.shape))
            pseudo_label = pseudo_label * p_target / p_model
            pseudo_label = (pseudo_label/pseudo_label.sum(dim=-1, keepdim=True))
#             prCyan('After scaling with DA, pseudo_label: {}, shape: {}'.format(pseudo_label, pseudo_label.shape))
            
        max_probs, targets_u = torch.max(pseudo_label, dim=-1) #hz confirmed, this does not change w/wo DA
        mask = max_probs.ge(args.threshold).float()#hz confirmed, this does not change w/wo DA
        
        unlabeledtrain_loss = (F.cross_entropy(logits_u_strongs, targets_u, reduction='none') * mask).mean()
        
        
        current_lambda_u = args.lambda_u_max * current_warmup #FixMatch algo did not use unlabeled loss rampup schedule
        
        args.writer.add_scalar('train/lambda_u', current_lambda_u, global_iteration_count)
        args.writer.add_scalar('train/current_warmup', current_warmup, global_iteration_count)
        
        

        repeated_labels = l_labels.repeat_interleave(args.mu) #[0, 1, 3, 2] ---> [0, 0, 1, 1, 3, 3, 2, 2]
#         print('true label: {}, repeated_labels: {}'.format(l_labels, repeated_labels))
#         print('pseudo_label: {}, max_probs: {}, targets_u: {}, mask: {}'.format(pseudo_label, max_probs, targets_u, mask))
        
#         print('targets_u: {}'.format(targets_u))
        
        relative_consistency_loss_mask_unlabeled = (repeated_labels == targets_u).float().unsqueeze(1)
#         print('relative_consistency_loss_mask_unlabeled: {}, shape: {}'.format(relative_consistency_loss_mask_unlabeled, relative_consistency_loss_mask_unlabeled.shape))
        
        count_mask_unlabeled = relative_consistency_loss_mask_unlabeled.reshape(args.labeledtrain_batchsize, args.mu).sum(dim=1) + 1e-8
#         print('count_mask_unlabeled: {}, shape: {}'.format(count_mask_unlabeled, count_mask_unlabeled.shape))
        
        # Reshape predicted_labels to group predictions for each image
        targets_u_reshaped = targets_u.reshape(args.labeledtrain_batchsize, args.mu)

        # 2. Indicator for each image if it has been predicted correctly at least once (size 5)
        # Broadcasting true_labels to compare with each group of predictions
        relative_consistency_loss_mask_labeled = (targets_u_reshaped == l_labels[:, None]).any(dim=1).int()
#         print('relative_consistency_loss_mask_labeled: {}, shape: {}'.format(relative_consistency_loss_mask_labeled, relative_consistency_loss_mask_labeled.shape))

        
        #add the new relative consistency loss
        labeled_diff = torch.softmax(logits_l_weak.detach()/args.FMLikeSharpening_T, dim=-1) - torch.softmax(logits_l_strong.detach()/args.FMLikeSharpening_T, dim=-1)
#         labeled_diff = logits_l_weak.softmax(1) - logits_l_strong.softmax(1)
        
#         print('before masking, logits_u_weaks:{}'.format(logits_u_weaks))
        logits_u_weaks = logits_u_weaks * relative_consistency_loss_mask_unlabeled
#         print('after masking, logits_u_weaks:{}'.format(logits_u_weaks))

        logits_u_strongs = logits_u_strongs * relative_consistency_loss_mask_unlabeled

        
        unlabeled_diff = logits_u_weaks.softmax(1).reshape(args.labeledtrain_batchsize, args.mu, args.num_classes).sum(dim=1) - logits_u_strongs.softmax(1).reshape(args.labeledtrain_batchsize, args.mu, args.num_classes).sum(dim=1)
                
#         print('labeled_diff: {}, shape: {}'.format(labeled_diff, labeled_diff.shape)) #--torch.Size([64, 10])
#         print('before average, unlabeled_diff: {}, shape: {}'.format(unlabeled_diff, unlabeled_diff.shape))#--torch.Size([64, 10])
        unlabeled_diff = unlabeled_diff/count_mask_unlabeled.unsqueeze(1)
#         print('after average, unlabeled_diff: {}, shape: {}'.format(unlabeled_diff, unlabeled_diff.shape))#--torch.Size([64, 10])

#         relative_loss = F.mse_loss(unlabeled_diff, labeled_diff.detach(), reduction='mean')
        relative_loss = (F.mse_loss(unlabeled_diff, labeled_diff.detach()) * relative_consistency_loss_mask_labeled).mean()
        
#         print('relative_loss: {}, shape: {}'.format(relative_loss, relative_loss.shape))
        current_lambda_relative_loss = args.lambda_relative_loss * current_relative_loss_warmup #FixMatch algo did not use unlabeled loss rampup schedule
        
        args.writer.add_scalar('train/lambda_relative_loss', current_lambda_relative_loss, global_iteration_count)
        args.writer.add_scalar('train/current_relative_loss_warmup', current_relative_loss_warmup, global_iteration_count)
        
    
        #Total loss
        loss = labeledtrain_loss + current_lambda_u * unlabeledtrain_loss + current_lambda_relative_loss * relative_loss
        
#         print('mask is {}'.format(mask))
        args.writer.add_scalar('train/gt_mask', mask.mean(), global_iteration_count)
        
        if args.em > 0:
            raise NameError('Need to think about how to use em regularization in FixMatch')
# #             loss -= args.em * ((combined_outputs.softmax(1) * F.log_softmax(combined_outputs, 1)).sum(1) * unlabeled_mask).mean()
#             loss -= args.em * ((logits_u.softmax(1) * F.log_softmax(logits_u, 1)).sum(1)).mean()
        
        ###############################################################################################################
        
        loss.backward()
        
        ######################################## Gradient clipping. HZ added 11/22/2023 ########################################
        #also reference https://github.com/TorchSSL/TorchSSL/blob/main/models/flexmatch/flexmatch.py#L196C21-L196C36
        if args.clip_norm >0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm) #after loss.backward() before optimizer.step()
        
        ########################################################################################################################
        
        
        MeanGradientNorm_this_iteration = np.mean([p.grad.data.norm(2).item() for p in model.parameters() if p.grad is not None])

        MeanGradientNorm.update(MeanGradientNorm_this_iteration)
        total_loss.update(loss.item())
        labeled_loss.update(labeledtrain_loss.item())
        unlabeled_loss_unscaled.update(unlabeledtrain_loss.item())
        unlabeled_loss_scaled.update(unlabeledtrain_loss.item() * current_lambda_u)
        relative_loss_unscaled.update(relative_loss.item())
        relative_loss_scaled.update(relative_loss.item() * current_lambda_relative_loss)
        mask_probs.update(mask.mean().item())

        
        MeanGradientNorm_this_epoch.append(MeanGradientNorm_this_iteration)
        TotalLoss_this_epoch.append(loss.item())
        LabeledLoss_this_epoch.append(labeledtrain_loss.item())
        UnlabeledLossUnscaled_this_epoch.append(unlabeledtrain_loss.item())
        UnlabeledLossScaled_this_epoch.append(unlabeledtrain_loss.item() * current_lambda_u)
        RelativeLossUnscaled_this_epoch.append(relative_loss.item())
        RelativeLossScaled_this_epoch.append(relative_loss.item() * current_lambda_relative_loss)
        
        optimizer.step()
        scheduler.step()
        
        #update ema model
        ema_model.update(model)
        
        model.zero_grad()
        
        
        batch_time.update(time.time() - end_time)
        
        #update end time
        end_time = time.time()


        #tqdm display for each minibatch update
        p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. MeanGradientNorm: {MeanGradientNorm:.3f}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {total_loss:.4f}. Loss_x: {labeled_loss:.4f}. Loss_u: {unlabeled_loss_unscaled:.4f}. Loss_relative: {relative_loss_unscaled:.4f}. Mask: {mask:.2f}. ".format(
                epoch=epoch + 1,
                epochs=args.train_epoch,
                batch=batch_idx + 1,
                MeanGradientNorm=MeanGradientNorm.avg,
                iter=args.iteration_per_epoch,
                lr=scheduler.get_last_lr()[0],
                data=data_time.avg,
                bt=batch_time.avg,
                total_loss=total_loss.avg,
                labeled_loss=labeled_loss.avg,
                unlabeled_loss_unscaled=unlabeled_loss_unscaled.avg,
                relative_loss_unscaled=relative_loss_unscaled.avg,
                mask=mask_probs.avg))
        p_bar.update()
        
        
        
##for debugging
#     print('fc.weight: {}'.format(model.fc.weight.cpu().detach().numpy()))
#     print('output.bias: {}'.format(model.output.bias.cpu().detach().numpy()))
        
#     print('ema fc.weight: {}'.format(ema_model.ema.fc.weight.cpu().detach().numpy()))
#     print('ema output.bias: {}'.format(ema_model.ema.output.bias.cpu().detach().numpy()))
        
    p_bar.close()
        
    MaxGradientNorm_this_epoch = np.max(MeanGradientNorm_this_epoch)
    
    return p_model, MaxGradientNorm_this_epoch, global_iteration_count, TotalLoss_this_epoch, LabeledLoss_this_epoch, UnlabeledLossUnscaled_this_epoch, UnlabeledLossScaled_this_epoch, RelativeLossUnscaled_this_epoch, RelativeLossScaled_this_epoch
        
   
    
    
    


#shared helper fct across different algos
def eval_model(args, data_loader, raw_model, ema_model, epoch, evaluation_criterion, weights=None):
    
    if evaluation_criterion == 'plain_accuracy':
        evaluation_method = calculate_plain_accuracy
    else:
        raise NameError('not supported yet')
    
    raw_model.eval()
    ema_model.eval()

    end_time = time.time()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    data_loader = tqdm(data_loader, disable=False)
    
    with torch.no_grad():
        total_targets = []
        total_raw_outputs = []
        total_ema_outputs = []
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            
            inputs = inputs.to(args.device).float()
            targets = targets.to(args.device).long()
            raw_outputs = raw_model(inputs)
            ema_outputs = ema_model(inputs)
            
            total_targets.append(targets.detach().cpu())
            total_raw_outputs.append(raw_outputs.detach().cpu())
            total_ema_outputs.append(ema_outputs.detach().cpu())
            
            if weights is not None:
                print('calculating weighted loss inside eval')
                loss = F.cross_entropy(raw_outputs, targets, weights)
            else:
                loss = F.cross_entropy(raw_outputs, targets)
            
            losses.update(loss.item(), inputs.shape[0])
            batch_time.update(time.time() - end_time)
            
            #update end time
            end_time = time.time()
            
            
        total_targets = np.concatenate(total_targets, axis=0)
        total_raw_outputs = np.concatenate(total_raw_outputs, axis=0)
        total_ema_outputs = np.concatenate(total_ema_outputs, axis=0)
        
        raw_performance = evaluation_method(total_raw_outputs, total_targets)
        ema_performance = evaluation_method(total_ema_outputs, total_targets)

        print('raw {} this evaluation step: {}'.format(evaluation_criterion, raw_performance), flush=True)
        print('ema {} this evaluation step: {}'.format(evaluation_criterion, ema_performance), flush=True)
        
        data_loader.close()
        
        
    return losses.avg, raw_performance, ema_performance, total_targets, total_raw_outputs, total_ema_outputs
    

#shared helper fct across different algos
def calculate_plain_accuracy(output, target):
    
    accuracy = (output.argmax(1) == target).mean()*100
    
    return accuracy


#shared helper fct across different algos
def save_pickle(save_dir, save_file_name, data):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    data_save_fullpath = os.path.join(save_dir, save_file_name)
    with open(data_save_fullpath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#shared helper fct across different algos
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


#shared helper fct across different algos
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
