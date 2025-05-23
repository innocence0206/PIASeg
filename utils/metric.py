import torch
import torch.nn.functional as F
import numpy as np
from medpy.metric import binary
from scipy.ndimage import label

def eval_dice(preds, labeval):
    assert preds.shape == labeval.shape
    
    axes = tuple(range(2, preds.ndim))
    TP = (preds * labeval).sum(axes)
    sum_pred = preds.sum(axes)
    sum_gt = labeval.sum(axes)
    summ = sum_pred + sum_gt
    
    eps = 1e-4
    dice = 2 * TP / (summ + eps)
    
    return dice.mean()

def eval_TPFPFN(preds, labeval):
    assert preds.shape == labeval.shape
    ones = torch.ones(preds.shape).to(preds.device)
    preds_ = ones - preds
    labeval_ = ones - labeval
    
    axes = tuple(range(2, preds.ndim))
    TP = (preds * labeval).sum(axes).float()
    FP = (preds * labeval_).sum(axes).float()
    FN = (preds_ * labeval).sum(axes).float()
    
    return TP.mean(), FP.mean(), FN.mean()

def eval_recall(preds, labeval):
    assert preds.shape == labeval.shape
    ones = torch.ones(preds.shape).to(preds.device)
    preds_ = ones - preds
    
    axes = tuple(range(2, preds.ndim))
    TP = (preds * labeval).sum(axes)
    FN = (preds_ * labeval).sum(axes)
    
    eps = 1e-4
    recall = TP / (TP + FN + eps)
    
    return recall.mean()

def eval_precision(preds, labeval):
    assert preds.shape == labeval.shape
    ones = torch.ones(preds.shape).to(preds.device)
    labeval_ = ones - labeval

    axes = tuple(range(2, preds.ndim))
    TP = (preds * labeval).sum(axes)
    FP = (preds * labeval_).sum(axes)
    
    eps = 1e-4
    precision = TP / (TP + FP + eps)
    
    return precision.mean()

def calculate_dice(pred, target):
    intersection= pred * target
    summ = pred + target
    
    intersection = intersection.sum().float()
    summ = summ.sum().float()
    
    eps = 1e-4
    dice = 2 * intersection / (summ + eps)

    return dice, intersection, summ

def compute_dice_coefficient(mask_gt, mask_pred):
  """Compute soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`. 
  
  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
  """
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum

def calculate_dice_split(pred, target, block_size=64*64*64):
    # evaluate every 128*128*128 block
    num = pred.shape[0]

    split_num = num // block_size     
    total_sum = 0
    total_intersection = 0
    
    for i in range(split_num):
        dice, intersection, summ = calculate_dice(pred[i*block_size:(i+1)*block_size], target[i*block_size:(i+1)*block_size])
        total_intersection += intersection
        total_sum += summ
    if num % block_size != 0:
        dice, intersection, summ = calculate_dice(pred[(i+1)*block_size:], target[(i+1)*block_size:])
        total_intersection += intersection
        total_sum += summ

    eps = 1e-4
    dice = 2 * total_intersection / (total_sum + eps)

    return dice, total_intersection, total_sum
  
def eval_metric(pred, lab):
    
    if np.sum(pred)==0 and np.sum(lab)==0:
        dice = 1.0
    else:
        dice = compute_dice_coefficient(pred, lab)

    if np.sum(pred)>=0 and np.sum(lab)==0:
        recall = 1.0
    else:
        recall = binary.recall(pred, lab)
    
    if np.sum(pred)==0 and np.sum(lab)>=0:
        precision = 1.0
    else:
        precision = binary.precision(pred, lab)
        
    return dice, recall, precision

def compute_hausdorffDist(pred, lab):
    pass
        
def ObjDice(pred, lab):
    
    # pred_dil = F.max_pool3d(torch.tensor(pred).float(), 3, 1, 2)
    
    pred_inst, _ = label(pred, structure=np.ones((3,3,3)))
    lab_inst, _ = label(lab, structure=np.ones((3,3,3)))

    # 获取pred中object列表及非0元素个数
    Pinst_list = np.unique(pred_inst)
    Pinst_list = Pinst_list[Pinst_list != 0]
    Pnum = len(Pinst_list)
    # 获取G中object列表及非0元素个数
    Linst_list = np.unique(lab_inst)
    Linst_list = Linst_list[Linst_list != 0]
    Lnum = len(Linst_list)

    if Pnum == 0 & Lnum == 0:
        return 1
    elif Pnum == 0 | Lnum == 0:
        return 0

    # 记录omega_i*Dice(G_i,S_i)
    temp1 = 0.0
    # pred中object总面积
    PtotalArea = np.sum(pred_inst > 0)

    for Pinst_i in range(Pnum):
        # Pi为P中值为Pinst_i的区域, boolean矩阵
        Pi1 = pred_inst == Pinst_list[Pinst_i]
        # 找到G中对应区域并去除背景
        intersectlist = lab_inst[Pi1]
        intersectlist = intersectlist[intersectlist != 0]

        if len(intersectlist) != 0:
            Lindex = np.argmax(np.bincount(intersectlist))
            # Li为gt中能与Pi面积重合最大的object
            Li1 = lab_inst == Lindex
        else:
            # Li = np.ones_like(lab_inst)
            # Li = Li == 0
            Li1 = np.zeros_like(lab_inst)

        Pomegai = np.sum(Pi1, dtype=np.float32) / PtotalArea
        temp1 = temp1 + Pomegai * compute_dice_coefficient(Li1, Pi1)

    # # 记录tilde_omega_i*Dice(tilde_G_i,tilde_S_i)
    temp2 = 0.0
    # G中object总面积
    LtotalArea = np.sum(lab_inst > 0)

    for Linst_i in range(Lnum):
        # Li为L中值为Linst_i的区域, boolean矩阵
        Li2 = lab_inst == Linst_list[Linst_i]
        # 找到P中对应区域并去除背景
        intersectlist = pred_inst[Li2]
        intersectlist = intersectlist[intersectlist != 0]

        if len(intersectlist) != 0:
            Pindex = np.argmax(np.bincount(intersectlist))
            Pi2 = pred_inst == Pindex
        else:
            # tildeSi = np.ones_like(S)
            # tildeSi = tildeSi == 0
            Pi2 = np.zeros_like(pred_inst)

        Lomegai = np.sum(Li2, dtype=np.float32) / LtotalArea
        temp2 = temp2 + Lomegai * compute_dice_coefficient(Li2, Pi2)

    objDice = (temp1 + temp2) / 2
    # objDice = temp2
    return objDice

def ObjHaus(pred, lab):
    pred_inst, _ = label(pred, structure=np.ones((3,3,3)))
    lab_inst, _ = label(lab, structure=np.ones((3,3,3)))

    # 获取pred中object列表及非0元素个数
    Pinst_list = np.unique(pred_inst)
    Pinst_list = Pinst_list[Pinst_list != 0]
    Pnum = len(Pinst_list)
    # 获取G中object列表及非0元素个数
    Linst_list = np.unique(lab_inst)
    Linst_list = Linst_list[Linst_list != 0]
    Lnum = len(Linst_list)
    
    PtotalArea = np.sum(pred_inst > 0)
    LtotalArea = np.sum(lab_inst > 0)

    temp1 = 0.0

    for Pinst_i in range(Pnum):
        # Pi为P中值为Pinst_i的区域, boolean矩阵
        Pi1 = pred_inst == Pinst_list[Pinst_i]
        # 找到G中对应区域并去除背景
        intersectlist = lab_inst[Pi1]
        intersectlist = intersectlist[intersectlist != 0]

        if len(intersectlist) != 0:
            Lindex = np.argmax(np.bincount(intersectlist))
            # Li为gt中能与Pi面积重合最大的object
            Li1 = lab_inst == Lindex
        else:
            # Li = np.ones_like(lab_inst)
            # Li = Li == 0
            Li1 = np.zeros_like(lab_inst)

        Pomegai = np.sum(Pi1, dtype=np.float32) / PtotalArea
        temp1 = temp1 + Pomegai * compute_hausdorffDist(Li1, Pi1)

    # 记录tilde_omega_i*Dice(tilde_G_i,tilde_S_i)
    temp2 = 0.0
    
    for Linst_i in range(Lnum):
        # Li为L中值为Linst_i的区域, boolean矩阵
        Li2 = lab_inst == Linst_list[Linst_i]
        # 找到P中对应区域并去除背景
        intersectlist = pred_inst[Li2]
        intersectlist = intersectlist[intersectlist != 0]

        if len(intersectlist) != 0:
            Pindex = np.argmax(np.bincount(intersectlist))
            Pi2 = pred_inst == Pindex
        else:
            # tildeSi = np.ones_like(S)
            # tildeSi = tildeSi == 0
            Pi2 = np.zeros_like(pred_inst)

        Lomegai = np.sum(Li2, dtype=np.float32) / LtotalArea
        temp2 = temp2 + Lomegai * compute_hausdorffDist(Li2, Pi2)

    objDice = (temp1 + temp2) / 2
    return objDice