import logging
import os
join=os.path.join
import math
from operator import mul
from functools import reduce
from PIL import Image
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch.distributed as dist

def is_master(args):
    return args.rank == 0

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class AverageMeter(object):
    """ Computes and stores the average and current value """

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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
        self.avg = self.sum / self.count if self.count != 0 else 0

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # logging.info("\t".join(entries))
        logging.info("  ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1)) 
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]" 
    

# LOG_FORMAT = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)s %(message)s"
LOG_FORMAT = "[%(levelname)s] %(asctime)s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

def configure_logger(rank, log_path=None):
    if log_path:
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)

    # only master process will print & write
    level = logging.INFO if rank in {-1, 0} else logging.WARNING
    handlers = [logging.StreamHandler()]
    if rank in {0, -1} and log_path:
        handlers.append(logging.FileHandler(log_path, "w"))

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=handlers,
        force=True,
    )

def save_configure(args):
    if hasattr(args, "distributed"):
        if (args.distributed and is_master(args)) or (not args.distributed):
            with open(f"{args.cp_dir}/config.txt", 'w') as f:
                for name in args.__dict__:
                    f.write(f"{name}: {getattr(args, name)}\n")
    else:
        with open(f"{args.cp_dir}/config.txt", 'w') as f:
            for name in args.__dict__:
                f.write(f"{name}: {getattr(args, name)}\n")

def resume_load_optimizer_checkpoint(optimizer, args):
    assert args.load != False, "Please specify the load path with --load"
    
    checkpoint = torch.load(args.load, map_location=torch.device('cpu'))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def resume_load_model_checkpoint(model, args, strict=True):
    assert args.load != False, "Please specify the load path with --load"
    
    checkpoint = torch.load(args.load, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    args.start_epoch = checkpoint['epoch']

def adjust_learning_rate(args, optimizer, epoch):
    if epoch >= 0 and epoch <= args.warmup_epoch and args.warmup_epoch != 0:
        lr = args.lr * 2.718 ** (10*(float(epoch) / float(args.warmup_epoch) - 1.))
        if epoch == args.warmup_epoch:
            lr = args.lr
    else:
        if args.lr_decay == 'poly':
            lr = args.lr * (1 - epoch / args.epochs)**0.9
        elif args.lr_decay == 'cosine':
            eta_min = lr * (args.lr_decay_rate**3)
            lr = (eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2)
        elif  args.lr_decay == 'step':
            steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
            if steps > 0:
                lr = lr * (args.lr_decay_rate**steps)
                
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    return lr

@torch.no_grad()
def concat_all_gather(tensor):
    """ 
    Performs all_gather operation on the provided tensor
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def new_lab3d(new_label, whole_labpath, crop_pos, args):
    whole_lab = np.load(whole_labpath, mmap_mode='c', allow_pickle=False).copy()
    whole_lab = whole_lab.astype(np.float32)
    pos = crop_pos.tolist()
    d, h, w = args.training_size
    whole_lab[pos[0]:pos[0]+d, pos[1]:pos[1]+h, pos[2]:pos[2]+w] = new_label
    
    return whole_lab

def save_result(pred, meta_info, args):
    pred = np.where(pred == 1, 255, 0)
    img = pred.astype(np.uint8)
    Img = Image.fromarray(img)
    
    case_save_path = join(args.test_preds, meta_info["casename"][0])
    os.makedirs(case_save_path, exist_ok=True)
    
    imgsave_path = join(case_save_path, meta_info["labname"][0].replace('full.npy','pred.png'))
    Img.save(imgsave_path)
    
def save_result_3d(pred, meta_info, args):
    # new_lab_path = join(args.test_preds, meta_info["casename"][0] + "_pred.npy")
    # np.save(new_lab_path, pred)
    
    pred = sitk.GetImageFromArray(pred)
    pred.SetSpacing(meta_info['spacing'][0].tolist())
    pred.SetDirection(meta_info['direction'][0].tolist())
    pred.SetOrigin(meta_info['origin'][0].tolist())

    save_path = join(args.test_preds, meta_info["casename"][0]+'_pred.nii.gz')
    sitk.WriteImage(pred, save_path)

def select_update(preds_meta, labori, labpse, eps, K):
    # preds_meta (B 1 H W); labori (B 1 H W); labpse (B 2 H W)
    B, Size = preds_meta.shape[0], preds_meta.shape[2:]
    
    preds = torch.sigmoid(preds_meta)
    if preds_meta.ndim == 4:
        lab_clean = F.max_pool2d(labori.float(), 5, 1, 2)  #kernel_size=5, stride=1, padding=2
    else:
        lab_clean = F.max_pool3d(labori.float(), 5, 1, 2)
        
    lab_corr = torch.where(labpse[:,1]==1, 1, 0)
    mask = ~(lab_clean.int() | lab_corr.unsqueeze(1)) + 2
    
    assert ((mask == 0).sum() + (mask == 1).sum()) == B * reduce(mul, Size)
    
    fg_prob = preds * mask
    fg_prob_ = fg_prob.squeeze(1).view(B,-1)
    _, indices = fg_prob_.topk(K, dim=1, largest=True, sorted=False)
    index_ = indices[:,None,:].expand(-1,2,-1)
    
    labels_seq = labpse.view(B, 2, -1).clone()
    need_corr = labels_seq.gather(2, index_) # B 2 K
    # need_corr和labels_seq是不共享内存的
    corr_meta = need_corr + eps
    
    bind = torch.cat([torch.ones(2*K, dtype=int)*int(i) for i in range(B)])
    cind = torch.cat([torch.cat((torch.zeros(K, dtype=int),torch.ones(K, dtype=int))) for _ in range(B)])
    kind = torch.LongTensor(index_.reshape(-1).cpu())
    index = (bind, cind, kind)
    
    labels_seq.index_put_(index, corr_meta.reshape(-1))
    labels_meta = labels_seq.view(B, 2, *Size)
    
    return labels_meta, index

def select_update_v2(preds_meta, labori, labpse, eps, FK, BK):
    # preds_meta (B 1 H W); labori (B 1 H W); labpse (B 2 H W)
    B, Size = preds_meta.shape[0], preds_meta.shape[2:]
    
    forepreds = torch.sigmoid(preds_meta)
    backpreds = 1 - forepreds
    # 0->1 correction
    if preds_meta.ndim == 4:
        lab_clean = F.max_pool2d(labori.float(), 5, 1, 2)  #kernel_size=5, stride=1, padding=2
    else:
        lab_clean = F.max_pool3d(labori.float(), 5, 1, 2)
        
    lab_corr = torch.where(labpse[:,1]==1, 1, 0)
    mask1_ = lab_clean.squeeze(1).int() | lab_corr
    mask1 = torch.where(mask1_==1, 0, 1)
    
    assert ((mask1 == 0).sum() + (mask1 == 1).sum()) == B * reduce(mul, Size)
    
    fg_prob = forepreds.squeeze(1) * mask1
    fg_prob_ = fg_prob.view(B,-1)
    _, fore_indices = fg_prob_.topk(FK, dim=1, largest=True, sorted=False)
    
    # 1->0 correction
    mask2 = torch.where(labpse[:,1]>0, 1, 0)

    bg_prob = backpreds.squeeze(1) * mask1 * mask2
    bg_prob_ = bg_prob.view(B,-1)
    _, back_indices = bg_prob_.topk(BK, dim=1, largest=True, sorted=False)
    
    indices = torch.cat((fore_indices, back_indices), dim=1)
    index_ = indices[:,None,:].expand(-1,2,-1)
    
    labels_seq = labpse.view(B, 2, -1).clone()
    need_corr = labels_seq.gather(2, index_) # B 2 K
    # need_corr和labels_seq是不共享内存的
    corr_meta = need_corr + eps
    
    bind = torch.cat([torch.ones(2*(FK+BK), dtype=int)*int(i) for i in range(B)])
    cind = torch.cat([torch.cat((torch.zeros((FK+BK), dtype=int),torch.ones((FK+BK), dtype=int))) for _ in range(B)])
    kind = torch.LongTensor(index_.reshape(-1).cpu())
    index = (bind, cind, kind)
    
    labels_seq.index_put_(index, corr_meta.reshape(-1))
    labels_meta = labels_seq.view(B, 2, *Size)
    
    return labels_meta, index, indices

def select_update_v3(preds_meta, labori, labpse, eps, FK, BK):
    # preds_meta (B 2 H W); labori (B 1 H W); labpse (B 2 H W)
    B, Size = preds_meta.shape[0], preds_meta.shape[2:]
    
    preds = torch.softmax(preds_meta, dim=1)
    backpreds, forepreds = preds[:,0], preds[:,1]

    # 0->1 correction
    if preds_meta.ndim == 4:
        lab_clean = F.max_pool2d(labori.float(), 5, 1, 2)  #kernel_size=5, stride=1, padding=2
    else:
        lab_clean = F.max_pool3d(labori.float(), 5, 1, 2)
        
    lab_corr = torch.where(labpse[:,1]==1, 1, 0)
    mask1_ = lab_clean.squeeze(1).int() | lab_corr
    mask1 = torch.where(mask1_==1, 0, 1)
    
    assert ((mask1 == 0).sum() + (mask1 == 1).sum()) == B * reduce(mul, Size)
    
    fg_prob = forepreds * mask1
    fg_prob_ = fg_prob.view(B,-1)
    _, fore_indices = fg_prob_.topk(FK, dim=1, largest=True, sorted=False)
    
    # 1->0 correction
    mask2 = torch.where(labpse[:,1]>0, 1, 0)
    bg_prob = backpreds * mask1 * mask2
    bg_prob_ = bg_prob.view(B,-1)
    _, back_indices = bg_prob_.topk(BK, dim=1, largest=False, sorted=False)
    
    indices = torch.cat((fore_indices, back_indices), dim=1)
    index_ = indices[:,None,:].expand(-1,2,-1)
    
    labels_seq = labpse.view(B, 2, -1).clone()
    need_corr = labels_seq.gather(2, index_) # B 2 K
    # need_corr和labels_seq是不共享内存的
    corr_meta = need_corr + eps
    
    bind = torch.cat([torch.ones(2*(FK+BK), dtype=int)*int(i) for i in range(B)])
    cind = torch.cat([torch.cat((torch.zeros((FK+BK), dtype=int),torch.ones((FK+BK), dtype=int))) for _ in range(B)])
    kind = torch.LongTensor(index_.reshape(-1).cpu())
    index = (bind, cind, kind)
    
    labels_seq.index_put_(index, corr_meta.reshape(-1))
    labels_meta = labels_seq.view(B, 2, *Size)
    
    return labels_meta, indices

def select_update_v4(preds_meta, labori, labpse, eps, FK, BK):
    # preds_meta (B 1 H W); labori (B 1 H W); labpse (B 2 H W)
    B, Size = preds_meta.shape[0], preds_meta.shape[2:]
    
    forepreds = torch.sigmoid(preds_meta)
    backpreds = 1 - forepreds
    # 0->1 correction
    lab_back = torch.where(labori==2, 1, 0)
    
    if preds_meta.ndim == 4:
        lab_clean = F.max_pool2d((labori==1).float(), 5, 1, 2)  #kernel_size=5, stride=1, padding=2
    else:
        lab_clean = F.max_pool3d((labori==1).float(), 5, 1, 2)
        
    lab_corr = torch.where(labpse[:,1]==1, 1, 0)
    
    mask1_ = lab_back.squeeze(1) | lab_clean.squeeze(1).int() | lab_corr
    mask1 = torch.where(mask1_==1, 0, 1)
    
    assert ((mask1 == 0).sum() + (mask1 == 1).sum()) == B * reduce(mul, Size)
    
    fg_prob = forepreds.squeeze(1) * mask1
    keep_mask = (fg_prob==0)
    fg_prob_ = fg_prob.view(B,-1)
    _, fore_indices = fg_prob_.topk(FK, dim=1, largest=True, sorted=False)
    
    # 1->0 correction
    mask2 = torch.where(labpse[:,1]>0, 1, 0)

    bg_prob = backpreds.squeeze(1) * mask1 * mask2
    bg_prob_ = bg_prob.view(B,-1)
    _, back_indices = bg_prob_.topk(BK, dim=1, largest=True, sorted=False)
    
    indices = torch.cat((fore_indices, back_indices), dim=1)
    index_ = indices[:,None,:].expand(-1,2,-1)
    
    labels_seq = labpse.view(B, 2, -1).clone()
    need_corr = labels_seq.gather(2, index_) # B 2 K
    # need_corr和labels_seq是不共享内存的
    corr_meta = need_corr + eps
    
    bind = torch.cat([torch.ones(2*(FK+BK), dtype=int)*int(i) for i in range(B)])
    cind = torch.cat([torch.cat((torch.zeros((FK+BK), dtype=int),torch.ones((FK+BK), dtype=int))) for _ in range(B)])
    kind = torch.LongTensor(index_.reshape(-1).cpu())
    index = (bind, cind, kind)
    
    labels_seq.index_put_(index, corr_meta.reshape(-1))
    labels_meta = labels_seq.view(B, 2, *Size)
    
    return labels_meta, indices, keep_mask

def select_update_noBK(preds_meta, labori, labpse, eps, FK):
    # preds_meta (B 1 H W); labori (B 1 H W); labpse (B 2 H W)
    B, Size = preds_meta.shape[0], preds_meta.shape[2:]
    
    forepreds = torch.sigmoid(preds_meta)
    backpreds = 1 - forepreds
    # 0->1 correction
    lab_back = torch.where(labori==2, 1, 0)
    
    if preds_meta.ndim == 4:
        lab_clean = F.max_pool2d((labori==1).float(), 5, 1, 2)  #kernel_size=5, stride=1, padding=2
    else:
        lab_clean = F.max_pool3d((labori==1).float(), 5, 1, 2)
        
    lab_corr = torch.where(labpse[:,1]==1, 1, 0)
    
    mask1_ = lab_back.squeeze(1) | lab_clean.squeeze(1).int() | lab_corr
    mask1 = torch.where(mask1_==1, 0, 1)
    
    assert ((mask1 == 0).sum() + (mask1 == 1).sum()) == B * reduce(mul, Size)
    
    fg_prob = forepreds.squeeze(1) * mask1
    keep_mask = (fg_prob==0)
    fg_prob_ = fg_prob.view(B,-1)
    _, indices = fg_prob_.topk(FK, dim=1, largest=True, sorted=False)
    
    index_ = indices[:,None,:].expand(-1,2,-1)
    
    labels_seq = labpse.view(B, 2, -1).clone()
    need_corr = labels_seq.gather(2, index_) # B 2 K
    # need_corr和labels_seq是不共享内存的
    corr_meta = need_corr + eps
    
    bind = torch.cat([torch.ones(2*FK, dtype=int)*int(i) for i in range(B)])
    cind = torch.cat([torch.cat((torch.zeros(FK, dtype=int),torch.ones(FK, dtype=int))) for _ in range(B)])
    kind = torch.LongTensor(index_.reshape(-1).cpu())
    index = (bind, cind, kind)
    
    labels_seq.index_put_(index, corr_meta.reshape(-1))
    labels_meta = labels_seq.view(B, 2, *Size)
    
    return labels_meta, indices, keep_mask