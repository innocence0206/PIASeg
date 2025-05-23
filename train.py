import logging
import os
import setproctitle # type: ignore
import time
import random
import yaml
import argparse
from tqdm import tqdm
import higher

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from utils.Dataset import *
from utils.train_utils import *
from utils.utils_loss import SoftCEDiceLoss, PriorContloss
from utils.metric import *
from model import Generic_UNetPri_L4
from inference3d import inference_sliding_window as inference

def train_net(model, trainset, suppset, valset, testset, args):
    
    trainLoader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                    num_workers=args.num_workers, drop_last=True)
    suppLoader = data.DataLoader(suppset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    validLoader = data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    testLoader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    logging.info(f"Created Dataset and DataLoader")
    
    writer = SummaryWriter(os.path.join(f"{args.log_path}", f"{args.unique_name}")) if is_master(args) else None

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    if args.resume:
        resume_load_optimizer_checkpoint(optimizer, args)

    criterion_seg = SoftCEDiceLoss()
    criterion_pcls = nn.BCEWithLogitsLoss()
    criterion_pdis = nn.MSELoss()
    criterion_cont = PriorContloss()
    
    criterion = {'criterion_seg': criterion_seg, 'criterion_pcls': criterion_pcls, 'criterion_pdis': criterion_pdis, 'criterion_cont': criterion_cont}
    
    ########################################################################################
    # Start training
    best_Dice = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
        cur_lr = adjust_learning_rate(args, optimizer, epoch)
        logging.info(f"Current lr: {cur_lr:.4e}")

        train_epoch(trainLoader, suppLoader, model, optimizer, epoch, criterion, writer, args)
        
        if epoch+1>(args.epochs-10):
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.cp_dir, f'{epoch+1}.pth'))
            
        torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.cp_dir, 'latest.pth' if epoch+1<args.epochs else 'final.pth'))
        
        if (epoch+1) % args.val_freq == 0 or (epoch+1>(args.epochs-20)):
            if args.dimension == '2d':
                val_Dice, val_caseDice, _ = evaluate_2d(validLoader, model, args)
                writer.add_scalar("val_sliceDice/AVG", val_Dice, epoch+1)
                writer.add_scalar("val_Dice/AVG", val_caseDice, epoch+1)
            else:
                val_Dice, _ = evaluate_3d(validLoader, model, args)
                writer.add_scalar("val_Dice/AVG", val_Dice, epoch+1)
        
            if val_Dice >= best_Dice:
                best_Dice = val_Dice
                
                # Save the checkpoint with best performance
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(args.cp_dir, 'best.pth'))

            logging.info("Evaluation Done")
            logging.info(f"Dice: {val_Dice:.4f}/Best Dice: {best_Dice:.4f}")
        
        if (epoch+1)==args.epochs:
            args.test_preds = join(args.cp_dir, 'test_seg')
            os.makedirs(args.test_preds, exist_ok=True)
            if args.dimension == '2d':
                test_Dice, test_caseDice, case_dice_dict = evaluate_2d(testLoader, model, args, prefix='test')
                writer.add_scalar("test_sliceDice/AVG", test_Dice, epoch+1)
                writer.add_scalar("test_Dice/AVG", test_caseDice, epoch+1)
            else:
                test_Dice, case_dice_dict = evaluate_3d(testLoader, model, args, prefix='test')
                writer.add_scalar("test_Dice/AVG", test_Dice, epoch+1)
    
    return best_Dice, test_Dice, case_dice_dict

def train_epoch(trainLoader, suppLoader, model, optimizer, epoch, criterion, writer, args):
    batch_time = AverageMeter("Time", ":4.2f")
    epoch_loss = AverageMeter("Loss", ":.2f")
    epoch_seg_loss = AverageMeter("SegLoss", ":.2f")
    epoch_pcls_loss = AverageMeter("PclsLoss", ":.2f")
    epoch_pdis_loss = AverageMeter("PdisLoss", ":.2f")
    epoch_fp_loss = AverageMeter("FPconLoss", ":.2f")
    epoch_bp_loss = AverageMeter("BPconLoss", ":.2f")

    epoch_corr2pos = AverageMeter("corr2pos", ":.2f")
    epoch_true_dice = AverageMeter("True_Dice", ":.4f")
    epoch_corr_dice = AverageMeter("Corr_Dice", ":.4f")
    epoch_corr_recall = AverageMeter("Corr_Recall", ":.4f")
    epoch_corr_precision = AverageMeter("Corr_Prec", ":.4f")
    epoch_cor_TP = AverageMeter("Cor_TP", ":.4f")
    epoch_cor_FP = AverageMeter("Cor_FP", ":.4f")
    epoch_cor_FN = AverageMeter("Cor_FN", ":.4f")

    progress = ProgressMeter(
        len(trainLoader),
        [batch_time, epoch_seg_loss, epoch_true_dice, epoch_corr_dice, epoch_corr_recall, epoch_corr_precision], 
        prefix="Epoch: [{}]".format(epoch+1),
    )
    
    model.train()
    ema_param = (1.0 * epoch / args.epochs * (args.rho_end_ - args.rho_start_) + args.rho_start_)
    
    if epoch % args.corr_rec_freq == 0:
        args.cur_save_dir = join(args.save_corrdir, str(epoch+args.corr_rec_freq).rjust(3,'0'))
        os.makedirs(args.cur_save_dir, exist_ok=True)
    
    start = time.time()
    for i, (img, labpse_, labori, labeval, meta_info) in enumerate(trainLoader):
        
        img, labpse_, labori, labeval = img.cuda(), labpse_.cuda(), labori.cuda(), labeval.cuda()
        labpse_ = torch.where(labpse_==2, 0, labpse_)
        labpse = torch.cat([1 - labpse_, labpse_], dim=1).float().detach()
        
        if epoch < args.corr_start:
            labels_final = labpse
        else:
            meta_model = create_model(args)
            meta_model.load_state_dict(model.state_dict())
            
            inner_opt = torch.optim.SGD(meta_model.parameters(), lr=0.001)

            # 使用 higher.innerloop_ctx 创建内循环上下文
            with higher.innerloop_ctx(meta_model, inner_opt, copy_initial_weights=True) as (fmodel, diffopt):
                # 内循环：在支持集上训练模型
                FPrior, BPrior = fmodel.priorlayer.fore_priors, fmodel.priorlayer.back_priors
                preds_meta, _, feat, _ = fmodel(img)
                
                B, C, Size = preds_meta.shape[0], feat.shape[1], preds_meta.shape[2:]
                N = reduce(mul, list(Size))
                FK = int(N * args.Fcorr_ratio)
                BK = int(N * args.Bcorr_ratio)
                eps = torch.zeros((B, 2, FK+BK), requires_grad=True).cuda()
                
                if BK > 0:
                    labels_meta, Kindices, keepmask = select_update_v4(preds_meta, labori, labpse, eps, FK, BK)
                else:
                    labels_meta, Kindices, keepmask = select_update_noBK(preds_meta, labori, labpse, eps, FK)
                
                l_meta = criterion['criterion_seg'](preds_meta, labels_meta)

                # 内循环梯度更新
                diffopt.step(l_meta)
                
                try:
                    img_supp, lab_supp_, _ = next(support_iter)
                except:
                    support_iter = iter(suppLoader)
                    img_supp, lab_supp_, _ = next(support_iter)

                img_supp = img_supp.cuda()
                lab_supp = torch.cat([1 - lab_supp_, lab_supp_], dim=1).cuda().float()

                preds_sup = fmodel(img_supp, 'test')
                l_sup = criterion['criterion_seg'](preds_sup, lab_supp)
                
                grad_eps = torch.autograd.grad(l_sup, eps, only_inputs=True, allow_unused=True)[0]

                eps = eps - grad_eps
                update_direction = eps.argmax(dim=1)
                epoch_corr2pos.update((update_direction==1).sum().item())

                if epoch < args.pcorr_start:
                    update_directions = labpse[:,1].view(B, -1).clone()
                    Bindices = torch.cat([torch.ones((FK+BK), dtype=int)*int(i) for i in range(B)])
                    update_directions.index_put_((Bindices,Kindices.reshape(-1)), update_direction.float().reshape(-1))
                    update_directions = update_directions.view(B, *Size)
                    cur_ema_param = ema_param
                else:
                    feat = feat.view(B, C, -1).permute(0, 2, 1).contiguous()
                    feat_corr = feat.gather(1, Kindices[:, :, None].expand(-1, -1, C))
                    # feat_corr = feat.gather(1, Kindices[:, :FK, None].expand(-1, -1, C))
                    priors = torch.cat((FPrior, BPrior), dim=0)
                    cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)

                    update_directions = []
                    update_amplitudes = []

                    for b in range(B):
                        sim_metric = cos_sim(feat_corr[b][:, None, :], priors[None, :, :])
                        sim_metric = sim_metric.argmax(dim=1)
                        sim_lab = torch.where(sim_metric < args.Fprior_num, 1, 0)  # K
                        # sim_lab = torch.cat((sim_lab, torch.zeros(BK, dtype=sim_lab.dtype, device=sim_lab.device)),dim=0)
                        same_mask = update_direction[b] == sim_lab
                        direction_filter = torch.masked_select(update_direction[b], same_mask)
                        ind_filter = torch.masked_select(Kindices[b], same_mask)
                        update_direction_i = labpse[b, 1].clone()
                        del sim_metric
                        torch.cuda.empty_cache()

                        if ind_filter.shape[0] > 0:
                            update_direction_i = update_direction_i.reshape(-1)
                            update_direction_i.index_put_((ind_filter,), direction_filter.float())
                            update_direction_i = update_direction_i.view(*Size)

                        update_amplitude_i = torch.ones(*Size).float().cuda()

                        update_directions.append(update_direction_i)
                        update_amplitudes.append(update_amplitude_i)

                    update_directions = torch.stack(update_directions)
                    update_amplitudes = torch.stack(update_amplitudes)
                    cur_ema_param = (ema_param * update_amplitudes).unsqueeze(1).expand(-1, 2, -1, -1, -1)

                update_directions = torch.where(keepmask, labpse_.squeeze(1), update_directions)
                
                corrlab = torch.stack([1 - update_directions, update_directions], dim=1)
                corrlab = corrlab.float().detach()
                
                for b in range(B):
                    if meta_info["labtype"][b] == 'full':
                        corrlab[:,1] = labpse_.squeeze(1).clone()
                        corrlab[:,0] = (1 - labpse_).squeeze(1).clone()
                
                updated_labels = labpse * (1 - cur_ema_param) + corrlab * cur_ema_param
                
                labels_final = updated_labels.detach()
                del grad_eps
        
        preds_final, prior_cls, preds_feat, prior_con = model(img)
        
        loss_seg = criterion['criterion_seg'](preds_final, labels_final)
        
        prior_lab = torch.cat((torch.zeros(args.Bprior_num, dtype=int), torch.ones(args.Fprior_num, dtype=int))).cuda().float()
        loss_pcls = criterion['criterion_pcls'](prior_cls, prior_lab.unsqueeze(0).expand(img.shape[0],-1))
        
        Fprior, Bprior = model.priorlayer.fore_priors, model.priorlayer.back_priors
        loss_Fpdis = -1 * criterion['criterion_pdis'](Fprior[:,None,:].expand(-1,args.Fprior_num,-1), Fprior[None,:,:].expand(args.Fprior_num,-1,-1))
        loss_Bpdis = -1 * criterion['criterion_pdis'](Bprior[:,None,:].expand(-1,args.Bprior_num,-1), Bprior[None,:,:].expand(args.Bprior_num,-1,-1))
        
        ############  Prototype Contrast Loss ###########
        loss_FPs, loss_BPs = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        for b in range(preds_final.shape[0]):
            Bprior_con, Fprior_con = prior_con[b, :args.Bprior_num], prior_con[b, args.Bprior_num:]
            Bprior_avg, Fprior_avg = Bprior_con.mean(dim=0, keepdim=True), Fprior_con.mean(dim=0, keepdim=True)
            
            if meta_info["labtype"][b] == 'full':
                neg_mask = torch.zeros((1, 1, *args.training_size)).cuda()
                z, y, x = torch.randint(0, args.training_size[0]-16, (1,)), torch.randint(0, args.training_size[1]-16, (1,)), torch.randint(0, args.training_size[2]-16, (1,))
                neg_mask[:,:,z:z+16, y:y+16, x:x+16] = 1
                neg_feat = F.adaptive_avg_pool3d(preds_feat*neg_mask, (1, 1, 1))
                # loss_BPs += criterion['criterion_cont'](neg_feat.squeeze((2,3,4)), Bprior_avg, Fprior_con)
                loss_BPs += criterion['criterion_cont'](neg_feat.squeeze((2,3,4)), Bprior_avg, Fprior)
            else:
                if (labori[b]==1).any():
                    fmask = labori[b]==1
                    pos_feat = F.adaptive_avg_pool3d(preds_feat*fmask, (1, 1, 1))
                    loss_FPs += criterion['criterion_cont'](pos_feat.squeeze((2,3,4)), Fprior_avg, Bprior_con)
                if (labori[b]==2).any():
                    bmask = labori[b]==2
                    neg_feat = F.adaptive_avg_pool3d(preds_feat*bmask, (1, 1, 1))
                    loss_BPs += criterion['criterion_cont'](neg_feat.squeeze((2,3,4)), Bprior_avg, Fprior_con)
        #################################################
        
        # loss = loss_seg + 0.9*loss_pcls + 0.01*(loss_Fpdis+loss_Bpdis)
        loss = loss_seg + 0.5*loss_pcls + (0.01*loss_FPs+0.02*loss_BPs) + 0.01*(loss_Fpdis+loss_Bpdis)
        # loss = loss_seg + 0.8*loss_pcls + (0.01*loss_FPs+0.01*loss_BPs) + 0.01*(loss_Fpdis+loss_Bpdis)
        
        epoch_loss.update(loss.item())
        epoch_seg_loss.update(loss_seg.item())
        epoch_pcls_loss.update(loss_pcls.item())
        epoch_pdis_loss.update((loss_Fpdis+loss_Bpdis).item())
        epoch_fp_loss.update(0.01*loss_FPs.item())
        epoch_bp_loss.update(0.02*loss_BPs.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch >= args.corr_start:
            new_label = torch.where(labori==2, 2, labels_final[:,1])
            new_label = new_label.float().cpu().numpy()

            for b in range(new_label.shape[0]):
                if meta_info["labtype"][b] == 'full':
                        continue
                if args.dimension == '2d':
                    case_lab_path = join(args.cur_save_dir, meta_info["casename"][b])
                    os.makedirs(case_lab_path, exist_ok=True)
                    new_lab_path = join(case_lab_path, meta_info["labname"][b])
                    np.save(new_lab_path, new_label[b])
                    trainLoader.dataset.updata_labels(meta_info["labpath_id"][b], new_lab_path)
                else:
                    complete_label = new_lab3d(new_label[b], meta_info["pselab_path"][b], meta_info["crop_para"][b], args)
                    new_lab_path = join(args.cur_save_dir, meta_info["casename"][b] + "_part.npy")
                    np.save(new_lab_path, complete_label)
                    trainLoader.dataset.updata_labels(meta_info["labpath_id"][b], new_lab_path)
                
        # if args.eval_corr:
        preds = torch.sigmoid(preds_final)
        pred_dice = eval_dice(preds, labeval)
        corr_dice = eval_dice(labels_final[:,1].unsqueeze(1), labeval)
        corr_recall = eval_recall(labels_final[:,1].unsqueeze(1), labeval)
        corr_precision = eval_precision(labels_final[:,1].unsqueeze(1), labeval)
        cor_TP, cor_FP, cor_FN = eval_TPFPFN(labels_final[:,1].unsqueeze(1), labeval)
        
        epoch_true_dice.update(pred_dice.item())
        epoch_corr_dice.update(corr_dice.item())
        epoch_corr_recall.update(corr_recall.item())
        epoch_corr_precision.update(corr_precision.item())
        epoch_cor_TP.update(cor_TP.item())
        epoch_cor_FP.update(cor_FP.item())
        epoch_cor_FN.update(cor_FN.item())

        batch_time.update(time.time() - start)
        start = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
            
        writer.add_scalar('Loss', epoch_loss.avg, epoch+1)
        writer.add_scalar('SegLoss', epoch_seg_loss.avg, epoch+1)
        writer.add_scalar('PclsLoss', epoch_pcls_loss.avg, epoch+1)
        # writer.add_scalar('FPdisLoss', epoch_FBpdis_loss.avg, epoch+1)
        writer.add_scalar('FPconLoss', epoch_fp_loss.avg, epoch + 1)
        writer.add_scalar('BPconLoss', epoch_bp_loss.avg, epoch + 1)
        writer.add_scalar('PdisLoss', epoch_pdis_loss.avg, epoch+1)
        
        writer.add_scalar('Train/Corr_Dice', epoch_corr_dice.avg, epoch+1)
        writer.add_scalar('Train/Corr_Recall', epoch_corr_recall.avg, epoch+1)
        writer.add_scalar('Train/Corr_Precision', epoch_corr_precision.avg, epoch+1)
        writer.add_scalar('Train/True_Dice', epoch_true_dice.avg, epoch+1)
        writer.add_scalar('Cor_TP', epoch_cor_TP.avg, epoch+1)
        writer.add_scalar('Cor_FP', epoch_cor_FP.avg, epoch+1)
        writer.add_scalar('Cor_FN', epoch_cor_FN.avg, epoch+1)
        writer.add_scalar('Corr_Pos', epoch_corr2pos.avg, epoch+1)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch+1)
        
        torch.cuda.empty_cache()

def evaluate_2d(Loader, model, args, prefix='val'):
    
    model.eval()
    logging.info(f"Evaluating")
    
    slice_dices = 0.0
    case_TP_dict, case_sum_dict, case_dice_dict = {}, {}, {}
    
    with torch.no_grad():
        iterator = tqdm(Loader)
        for (img, lab, meta_info) in iterator:
            casename = meta_info['casename'][0]
            img, lab = img.cuda(), lab.cuda()
            
            output, _, _, _ = model(img)
            pred = torch.sigmoid(output)

            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0

            pred = pred.to(torch.int8)
            del img
            
            slice_dice, TP, sum = calculate_dice(pred.squeeze().reshape(-1), lab.squeeze().reshape(-1))
            
            slice_dices += slice_dice
            if casename in case_TP_dict:
                case_TP_dict[casename] += TP
                case_sum_dict[casename] += sum
            else:
                case_TP_dict[casename], case_sum_dict[casename] = TP, sum

            if prefix == 'test':
                save_result(pred.squeeze().cpu().numpy(), meta_info, args)
            
            del pred, lab
            torch.cuda.empty_cache()
    
    slice_dices /= len(Loader)
    for case_eval in case_TP_dict:
        dice = 2 * case_TP_dict[case_eval] / case_sum_dict[case_eval]
        case_dice_dict[case_eval] = dice.item()

    dice_list = list(case_dice_dict.values())
    dice_mean = np.array(dice_list).mean()

    return slice_dices, dice_mean, case_dice_dict

def evaluate_3d(Loader, model, args, prefix='val'):
    
    model.eval()
    logging.info(f"Evaluating")
    
    case_dice_dict = {}
    dice_list = []
    
    with torch.no_grad():
        iterator = tqdm(Loader)
        for (img, lab, meta_info) in iterator:
            img, lab = img.cuda(), lab.cuda()
    
            B, _, D, H, W = img.shape

            z_len = 600.

            if D > z_len:
                num_z_chunks = math.ceil(D / z_len)
                z_chunk_len = math.ceil(D / num_z_chunks)

                image_chunk_list = []
                for i in range(num_z_chunks):
                    image_chunk_list.append(img[:, :, i*z_chunk_len:(i+1)*z_chunk_len, :, :])
                label_pred_list = []
                for image_chunk in image_chunk_list:
                    pred = inference(model, image_chunk, args)
                    label_pred_list.append(pred)
                pred = torch.cat(label_pred_list, dim=2)

            else:
                pred = inference(model, img, args)

            del img
            torch.cuda.empty_cache()

            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            pred = pred.to(torch.int8)
            torch.cuda.empty_cache()
            
            pred = pred.squeeze(0)
            lab = lab.squeeze(0)
            
            if prefix=='test' and args.save_pred:
                save_result_3d(pred, meta_info, args)
            
            torch.cuda.empty_cache()
            tmp_dice, _, _ = calculate_dice_split(pred.reshape(-1), lab.reshape(-1))

            del pred, lab
            torch.cuda.empty_cache()

            tmp_dice = tmp_dice.unsqueeze(0)
        
            if args.distributed:
                # gather results from all gpus
                tmp_dice = concat_all_gather(tmp_dice)
                
            tmp_dice = tmp_dice.cpu().numpy()
            for idx in range(len(tmp_dice)):  # get the result for each sample
                dice_list.append(tmp_dice[idx])
    
    # Due to the DistributedSampler pad samples to make data evenly distributed to all gpus,
    # we need to remove the padded samples for correct evaluation.
    if args.distributed:
        world_size = dist.get_world_size()
        dataset_len = len(Loader.dataset)

        padding_size = 0 if (dataset_len % world_size) == 0 else world_size - (dataset_len % world_size)
        
        for _ in range(padding_size):
            dice_list.pop()
            
    dice_mean = np.array(dice_list).mean()

    return dice_mean, case_dice_dict

def get_parser():
    parser = argparse.ArgumentParser(description='Partial Instance Annotations medical image segmentation')
    parser.add_argument('--dataset', type=str, default='Covid1920', help='dataset name')
    parser.add_argument('--model', type=str, default='pianet', help='model name')
    parser.add_argument('--dimension', type=str, default='2d', help='2d model or 3d model')

    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--resume', action='store_true', help='if resume training from checkpoint')
    parser.add_argument('--load', type=str, default=False, help='load pretrained model')
    parser.add_argument('--cp_path', type=str, default='./exp', help='the path to save checkpoint and logging info')
    parser.add_argument('--log_path', type=str, default='./log', help='the path to save tensorboard log')
    parser.add_argument('--unique_name', type=str, default='test', help='unique experiment name')
    parser.add_argument('--save_pred', action='store_true', help='save test results')
    parser.add_argument('--gpu', type=str, default='2')
    args = parser.parse_args()
    
    config_path = 'config/%s_%s.yaml'%(args.dataset, args.dimension)
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s"%config_path)

    logging.info('Loading configurations from %s'%config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)
    
    args.cp_dir = f"{args.cp_path}/{args.dataset}/{args.unique_name}"
    os.makedirs(args.cp_dir, exist_ok=True)
    args.save_corrdir = join(args.cp_dir, "corr_labels")
    os.makedirs(args.save_corrdir, exist_ok=True)

    return args

def create_model(args):
    
    if args.dataset == 'Covid1920_1p5':
        model = Generic_UNetPri_L4(in_channels=args.in_channels, FPnum=args.Fprior_num,BPnum=args.Bprior_num).cuda()
    
    elif args.dataset == 'ISLES22':
        model = Generic_UNetPri_L4(in_channels=args.in_channels, FPnum=args.Fprior_num,BPnum=args.Bprior_num).cuda()

    elif args.dataset == 'MosMed':
        model = Generic_UNetPri_L4(in_channels=args.in_channels, FPnum=args.Fprior_num,BPnum=args.Bprior_num).cuda()
    
    else:
        model = Generic_UNetPri_L4(in_channels=args.in_channels, FPnum=args.Fprior_num,BPnum=args.Bprior_num).cuda()
        
    if args.resume:
        resume_load_model_checkpoint(model, args)
        
    return model

def main_worker(proc_idx, ngpus_per_node, args, trainset=None, suppset=None, valset=None, testset=None, result_dict=None):
    # seed each process
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if hasattr(torch, "set_deterministic"):
            torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
    args.proc_idx = proc_idx
    args.ngpus_per_node = ngpus_per_node
    
    configure_logger(args.rank, args.cp_dir+f"/log.txt")
    save_configure(args)
    
    logging.info(
        f"\nDataset: {args.dataset},\n"
        + f"Model: {args.model},\n"
        + f"Dimension: {args.dimension}"
    )
    
    model = create_model(args)

    model.to(f"cuda")

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.proc_idx], find_unused_parameters=True)
        # set find_unused_parameters to True if some of the parameters is not used in forward

    logging.info(f"Created Model")
    best_Dice, test_Dice, case_dice_dict = train_net(model, trainset, suppset, valset, testset, args)
    
    logging.info(f"Training and evaluation are done")
    
    if args.distributed:
        if is_master(args):
            # collect results from the master process
            pass
    else:
        return best_Dice, test_Dice, case_dice_dict

if __name__ == '__main__':
    setproctitle.setproctitle("它要占用 13000 MiB!!!! ")
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.log_path = os.path.join(args.log_path, args.dataset)
    
    ngpus_per_node = torch.cuda.device_count()
    
    args.multiprocessing_distributed = False
    
    # args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.distributed = args.multiprocessing_distributed
    
    if args.multiprocessing_distributed:
        pass
    else:
        if args.dataset not in ['ISLES22_1p5', 'ISLES22']:
            trainset, suppset = Dataset3D(args, mode='train'), Dataset3D(args, mode='supp')
            valset, testset = Dataset3D(args, mode='val'), Dataset3D(args, mode='test')
        else:
            trainset, suppset = ISLES223D(args, mode='train'), ISLES223D(args, mode='supp')
            valset, testset = ISLES223D(args, mode='val'), ISLES223D(args, mode='test')
        
        best_Dice, test_Dice, case_dice_dict = main_worker(0, ngpus_per_node, args, trainset, suppset, valset, testset)
        
    with open(f"{args.cp_path}/{args.dataset}/{args.unique_name}/results.txt",  'w') as f:
        np.set_printoptions(precision=4, suppress=True) 
        f.write(f'VAL Best Dice: {best_Dice}\n')
        f.write(f'Test Dice: {test_Dice}\n')
        f.write(f"Each Case Dice: \n")
        for name, value in case_dice_dict.items():
            f.write(f"{name} : {value}\n")
        
    logging.info('Training done.')