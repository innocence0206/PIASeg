import os
join=os.path.join
import glob
import json
import math
import copy
import torch
from PIL import Image
import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset
from . import augmentation

class Dataset2D(Dataset):
    def __init__(self, args, mode='train'):
        assert mode in ['train', 'supp', 'val', 'test']
        self.args = args
        self.mode = mode
        data_path = join(args.data_root, args.dataset)
        self._set_file_paths(data_path, args.split_path)
        
        print('All', mode, 'data load done, length of', mode, 'set:', len(self.img_paths))
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):

        np_img = np.load(self.img_paths[index], mmap_mode='r', allow_pickle=False)
        tensor_img = torch.from_numpy(np_img.copy()).unsqueeze(0).unsqueeze(0).float()
        # 1, C, H, W
        if self.mode != 'train':
            lab_path = self.fullab_paths[index]
            np_fullab = np.load(lab_path, mmap_mode='r', allow_pickle=False)
            tensor_lab = torch.from_numpy(np_fullab.copy()).unsqueeze(0).unsqueeze(0)
        else:
            lab_path = self.pselab_paths[index]
            np_pselab = np.load(lab_path, mmap_mode='r', allow_pickle=False)
            tensor_lab = torch.from_numpy(np_pselab.copy()).unsqueeze(0).unsqueeze(0).float()

            np_partlab = np.load(self.partlab_paths[index], mmap_mode='r', allow_pickle=False)
            tensor_labori = torch.from_numpy(np_partlab.copy()).unsqueeze(0).unsqueeze(0)
            if self.args.eval_corr:
                np_fullab = np.load(self.fullab_paths[index], mmap_mode='r', allow_pickle=False)
                tensor_labeval = torch.from_numpy(np_fullab.copy()).unsqueeze(0).unsqueeze(0)
            
        # casename, labname = lab_path.split('/')[-2:]
        labname = os.path.basename(lab_path)
        casename = '_'.join(i for i in labname.split('_')[:2])
        
        if self.mode == 'train':
            if np.random.random() < self.args.color_prob:
                tensor_img = augmentation.brightness_multiply(tensor_img, multiply_range=[0.7, 1.3])
            if np.random.random() < self.args.color_prob:
                tensor_img = augmentation.brightness_additive(tensor_img, std=0.1)
            if np.random.random() < self.args.color_prob:
                tensor_img = augmentation.gamma(tensor_img, gamma_range=[0.7, 1.5])
            if np.random.random() < self.args.color_prob:
                tensor_img = augmentation.contrast(tensor_img, contrast_range=[0.7, 1.3])
            if np.random.random() < self.args.color_prob:
                tensor_img = augmentation.gaussian_blur(tensor_img, sigma_range=[0.5, 1.0])
            if np.random.random() < self.args.color_prob:
                std = np.random.random() * 0.1
                tensor_img = augmentation.gaussian_noise(tensor_img, std=std)

        tensor_img = augmentation.standarize(tensor_img)
        
        tensor_img = tensor_img.squeeze(0)
        tensor_lab = tensor_lab.squeeze(0)
        assert tensor_img.shape == tensor_lab.shape

        meta_info = {"casename": casename, "labname": labname, "labtype": labname[-8:-4], "labpath_id": index}
        if self.mode != "train":
            return (tensor_img.float(), tensor_lab.to(torch.int8), meta_info)
            
        else:
            tensor_labori = tensor_labori.squeeze(0)
            assert tensor_labori.shape == tensor_lab.shape
            
            if self.args.eval_corr:
                tensor_labeval = tensor_labeval.squeeze(0)
                assert tensor_labeval.shape == tensor_lab.shape
            else:
                tensor_labeval = torch.ones_like(tensor_lab) * (-1)

            return (tensor_img.float(), tensor_lab.float(), tensor_labori.to(torch.int8), tensor_labeval.to(torch.int8), meta_info)

    
    def updata_labels(self, index, new_path):
        self.pselab_paths[index] = new_path
        
    def _set_file_paths(self, path, split_path):
        self.img_paths = []
        self.partlab_paths = []
        self.fullab_paths = []
        split_info = json.load(open(split_path))
        
        if self.mode == 'train':
            # imgs = sorted(glob.glob(join(path, 'IMG_norm', '*.npy')))
            # self.img_paths.extend(imgs)
            # labs = sorted(glob.glob(join(path, 'GT_norm', '*.npy')))
            # self.partlab_paths.extend(labs)
            # self.fullab_paths.extend(labs)
            
            for sample in split_info["train"]:
                imgs = sorted(glob.glob(join(path, 'IMG', sample, '*.npy')))
                self.img_paths.extend(imgs)
                
                partlabs = sorted(glob.glob(join(path, 'GT_part', sample, '*.npy')))
                self.partlab_paths.extend(partlabs)
                if self.args.eval_corr:
                    fullabs = sorted(glob.glob(join(path, 'GT_full', sample, '*.npy')))
                    self.fullab_paths.extend(fullabs)
                        
            self.pselab_paths = copy.deepcopy(self.partlab_paths)
        # elif self.mode == 'supp':
        #     imgs = sorted(glob.glob(join(path, 'IMG_norm', '*.npy')))
        #     self.img_paths.extend(imgs)
        #     labs = sorted(glob.glob(join(path, 'GT_norm', '*.npy')))
        #     self.fullab_paths.extend(labs)
            # for sample in split_info[self.mode]:
            #     imgs = sorted(glob.glob(join('/mnt/data/gxy/PIA_Data/npy/2D/Covid1920/', 'IMG', sample, '*.npy')))
            #     self.img_paths.extend(imgs)
            #     labs = sorted(glob.glob(join('/mnt/data/gxy/PIA_Data/npy/2D/Covid1920/', 'GT_full', sample, '*.npy')))
            #     self.fullab_paths.extend(labs)
        else:
            for sample in split_info[self.mode]:
                imgs = sorted(glob.glob(join(path, 'IMG', sample, '*.npy')))
                self.img_paths.extend(imgs)
                labs = sorted(glob.glob(join(path, 'GT_full', sample, '*.npy')))
                self.fullab_paths.extend(labs)

class Dataset3D(Dataset):
    def __init__(self, args, mode='train'):
        assert mode in ['train', 'supp', 'val', 'test']
        self.args = args
        self.mode = mode
        data_path = join(args.data_root, args.dataset)
        self._set_file_paths(data_path, args.split_path)
        
        print('All', mode, 'data load done, length of', mode, 'set:', len(self.img_paths))
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):

        np_img = np.load(self.img_paths[index], mmap_mode='r', allow_pickle=False)
        tensor_img = torch.from_numpy(np_img.copy()).unsqueeze(0).unsqueeze(0).float()
        # 1, C, D, H, W
        if self.mode == 'train':
            lab_path = self.pselab_paths[index]
            labtype = os.path.basename(lab_path)[-8:-4]
            np_pselab = np.load(lab_path, mmap_mode='r', allow_pickle=False)
            tensor_lab = torch.from_numpy(np_pselab.copy()).unsqueeze(0).unsqueeze(0).float()
            
            np_partlab = np.load(self.partlab_paths[index], mmap_mode='r', allow_pickle=False)
            tensor_labori = torch.from_numpy(np_partlab.copy()).unsqueeze(0).unsqueeze(0)
            
            tensor_img, tensor_lab, tensor_labori, crop_para = augmentation.crop_3d_v3(tensor_img, tensor_lab, tensor_labori, self.args.training_size, mode='random')
            
            if self.args.eval_corr:
                np_fullab = np.load(self.fullab_paths[index], mmap_mode='r', allow_pickle=False)
                tensor_labeval = torch.from_numpy(np_fullab.copy()).unsqueeze(0).unsqueeze(0)
                tensor_labeval = augmentation.crop_3d_para(tensor_labeval, crop_para, self.args.training_size)
            
        elif self.mode == 'supp':
            lab_path = self.fullab_paths[index]
            np_fullab = np.load(lab_path, mmap_mode='r', allow_pickle=False)
            tensor_lab = torch.from_numpy(np_fullab.copy()).unsqueeze(0).unsqueeze(0)
            
            tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='random')
        else:
            lab_path = self.fullab_paths[index]
            np_fullab = np.load(lab_path, mmap_mode='r', allow_pickle=False)
            tensor_lab = torch.from_numpy(np_fullab.copy()).unsqueeze(0).unsqueeze(0)
                
        casename = os.path.basename(self.img_paths[index]).split('.npy')[0]
        
        if self.mode == 'train':
            if np.random.random() < self.args.color_prob:
                tensor_img = augmentation.brightness_multiply(tensor_img, multiply_range=[0.7, 1.3])
            if np.random.random() < self.args.color_prob:
                tensor_img = augmentation.brightness_additive(tensor_img, std=0.1)
            if np.random.random() < self.args.color_prob:
                tensor_img = augmentation.gamma(tensor_img, gamma_range=[0.7, 1.5])
            if np.random.random() < self.args.color_prob:
                tensor_img = augmentation.contrast(tensor_img, contrast_range=[0.7, 1.3])
            if np.random.random() < self.args.color_prob:
                tensor_img = augmentation.gaussian_blur(tensor_img, sigma_range=[0.5, 1.0])
            if np.random.random() < self.args.color_prob:
                std = np.random.random() * 0.1
                tensor_img = augmentation.gaussian_noise(tensor_img, std=std)
                
        tensor_img = tensor_img.squeeze(0)
        tensor_lab = tensor_lab.squeeze(0)
        
        assert tensor_img.shape == tensor_lab.shape

        if self.mode in ['supp', 'val', 'test']:
            meta_info = {"casename": casename}
                     
            return (tensor_img.float(), tensor_lab.to(torch.int8), meta_info)
        else:
            meta_info = {"casename": casename,
                         "labtype": labtype,
                         "pselab_path": self.pselab_paths[index],
                         "crop_para": torch.Tensor(crop_para).to(torch.int16), 
                         "labpath_id": index}
            
            tensor_labori = tensor_labori.squeeze(0)
            assert tensor_labori.shape == tensor_lab.shape
            
            if self.args.eval_corr:
                tensor_labeval = tensor_labeval.squeeze(0)
                assert tensor_labeval.shape == tensor_lab.shape
            else:
                tensor_labeval = torch.ones_like(tensor_lab) * (-1)

            return (tensor_img.float(), tensor_lab.float(), tensor_labori.to(torch.int8), tensor_labeval.to(torch.int8), meta_info)
    
    def updata_labels(self, index, new_path):
        self.pselab_paths[index] = new_path
        
    def _set_file_paths(self, path, split_path):
        self.img_paths = []
        self.partlab_paths = []
        self.fullab_paths = []
        split_info = json.load(open(split_path))
        
        if self.mode == 'train':
            imgs = sorted(glob.glob(join(path, 'IMG_norm', '*.npy')))
            self.img_paths.extend(imgs)
            labs = sorted(glob.glob(join(path, 'GT_norm', '*.npy')))
            self.partlab_paths.extend(labs)
            self.fullab_paths.extend(labs)
            
            for sample in split_info["train"]:
                self.img_paths.append(join(path, 'IMG', sample+'.npy'))
                self.partlab_paths.append(join(path, 'GT_pasc1', sample+'_part.npy'))
                if self.args.eval_corr:
                    self.fullab_paths.append(join(path, 'GT_full', sample+'_full.npy'))

            self.pselab_paths = copy.deepcopy(self.partlab_paths)
            
        else:
            for sample in split_info[self.mode]:
                self.img_paths.append(join(path, 'IMG', sample+'.npy'))
                self.fullab_paths.append(join(path, 'GT_full', sample+'_full.npy'))

class ISLES223D(Dataset):
    def __init__(self, args, mode='train'):
        assert mode in ['train', 'supp', 'val', 'test']
        self.args = args
        self.mode = mode
        data_path = join(args.data_root, args.dataset)
        self._set_file_paths(data_path, args.split_path)
        
        print('All', mode, 'data load done, length of', mode, 'set:', len(self.img_paths))
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        
        np_img = np.load(self.img_paths[index], mmap_mode='r', allow_pickle=False)
        tensor_img = torch.from_numpy(np_img.copy()).unsqueeze(0).float()
        # 1, C, D, H, W
        if self.mode == 'train':
            lab_path = self.pselab_paths[index]
            labtype = os.path.basename(lab_path)[-8:-4]
            np_pselab = np.load(lab_path, mmap_mode='r', allow_pickle=False)
            tensor_lab = torch.from_numpy(np_pselab.copy()).unsqueeze(0).unsqueeze(0).float()
            
            np_partlab = np.load(self.partlab_paths[index], mmap_mode='r', allow_pickle=False)
            tensor_labori = torch.from_numpy(np_partlab.copy()).unsqueeze(0).unsqueeze(0)
            
            tensor_img, tensor_lab, tensor_labori, crop_para = augmentation.crop_3d_v3(tensor_img, tensor_lab, tensor_labori, self.args.training_size, mode='random')
            
            if self.args.eval_corr:
                np_fullab = np.load(self.fullab_paths[index], mmap_mode='r', allow_pickle=False)
                tensor_labeval = torch.from_numpy(np_fullab.copy()).unsqueeze(0).unsqueeze(0)
                tensor_labeval = augmentation.crop_3d_para(tensor_labeval, crop_para, self.args.training_size)
        
        elif self.mode == 'supp':
            lab_path = self.fullab_paths[index]
            np_fullab = np.load(lab_path, mmap_mode='r', allow_pickle=False)
            tensor_lab = torch.from_numpy(np_fullab.copy()).unsqueeze(0).unsqueeze(0)
            
            tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='random')
        else:
            lab_path = self.fullab_paths[index]
            np_fullab = np.load(lab_path, mmap_mode='r', allow_pickle=False)
            tensor_lab = torch.from_numpy(np_fullab.copy()).unsqueeze(0).unsqueeze(0)
                
        casename = os.path.basename(self.img_paths[index]).split('.npy')[0]
        
        if self.mode == 'train':
            if np.random.random() < self.args.color_prob:
                tensor_img = augmentation.brightness_multiply(tensor_img, multiply_range=[0.7, 1.3], per_channel=True)
            if np.random.random() < self.args.color_prob:
                tensor_img = augmentation.brightness_additive(tensor_img, std=0.1, per_channel=True)
            if np.random.random() < self.args.color_prob:
                tensor_img = augmentation.gamma(tensor_img, gamma_range=[0.7, 1.5], per_channel=True)
            if np.random.random() < self.args.color_prob:
                tensor_img = augmentation.contrast(tensor_img, contrast_range=[0.7, 1.3], per_channel=True)
            # if np.random.random() < self.args.color_prob:
            #     tensor_img = augmentation.gaussian_blur(tensor_img, sigma_range=[0.5, 1.0])
            if np.random.random() < self.args.color_prob:
                std = np.random.random() * 0.1
                tensor_img = augmentation.gaussian_noise(tensor_img, std=std)
                
        tensor_img = tensor_img.squeeze(0)
        tensor_lab = tensor_lab.squeeze(0)
        
        assert tensor_img.shape[1:] == tensor_lab.shape[1:]

        
        if self.mode in ['supp', 'val', 'test']:
            meta_info = {"casename": casename}
                     
            return (tensor_img.float(), tensor_lab.to(torch.int8), meta_info)
            
        else:
            meta_info = {"casename": casename,
                         "labtype": labtype,
                         "pselab_path": self.pselab_paths[index],
                         "crop_para": torch.Tensor(crop_para).to(torch.int16), 
                         "labpath_id": index}
            
            tensor_labori = tensor_labori.squeeze(0)
            assert tensor_labori.shape == tensor_lab.shape
            
            if self.args.eval_corr:
                tensor_labeval = tensor_labeval.squeeze(0)
                assert tensor_labeval.shape == tensor_lab.shape
            else:
                tensor_labeval = torch.ones_like(tensor_lab) * (-1)

            return (tensor_img.float(), tensor_lab.float(), tensor_labori.to(torch.int8), tensor_labeval.to(torch.int8), meta_info)
    
    def updata_labels(self, index, new_path):
        self.pselab_paths[index] = new_path
                
    def _set_file_paths(self, path, split_path):
        self.img_paths = []
        self.partlab_paths = []
        self.fullab_paths = []
        split_info = json.load(open(split_path))
        
        if self.mode == 'train':
            imgs = sorted(glob.glob(join(path, 'IMG_norm', '*.npy')))
            self.img_paths.extend(imgs)
            labs = sorted(glob.glob(join(path, 'GT_norm', '*.npy')))
            self.partlab_paths.extend(labs)
            self.fullab_paths.extend(labs)
            
            for sample in split_info["train"]:
                self.img_paths.append(join(path, 'IMG', sample+'.npy'))
                self.partlab_paths.append(join(path, 'GT_pasc1', sample+'_part.npy'))
                if self.args.eval_corr:
                    self.fullab_paths.append(join(path, 'GT_full', sample+'_full.npy'))

            self.pselab_paths = copy.deepcopy(self.partlab_paths)

        else:
            for sample in split_info[self.mode]:
                self.img_paths.append(join(path, 'IMG', sample+'.npy'))
                self.fullab_paths.append(join(path, 'GT_full', sample+'_full.npy'))

class DatasetInfer2D(Dataset):
    def __init__(self, args):
        self.args = args
        data_path = join(args.infer_root, args.InferSet)
        self._set_file_paths(data_path, args.split_path)
        
        print('All data load done, length of images:', len(self.img_paths))
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        
        img = Image.open(self.img_paths[index])
        np_img = np.array(img)
        tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0).float()
        tensor_img = augmentation.standarize(tensor_img)
        
        lab = Image.open(self.lab_paths[index])
        np_lab = np.array(lab)
        np_lab = np.where(np_lab > 0, 1, 0)
        tensor_lab = torch.from_numpy(np_lab).unsqueeze(0)

        labname = os.path.basename(self.lab_paths[index])
        casename = '_'.join(i for i in labname.split('_')[:2])

        tensor_img = tensor_img.squeeze(0)
        assert tensor_img.shape == tensor_lab.shape

        meta_info = {"casename": casename, "labname": labname}
        
        return (tensor_img.float(), tensor_lab.to(torch.int8), meta_info)

    def _set_file_paths(self, path, split_path):
        self.img_paths, self.lab_paths = [], []
        split_info = json.load(open(split_path))

        for sample in split_info['test']:
            imgs = sorted(glob.glob(join(path, 'IMG', sample, '*.png')))
            self.img_paths.extend(imgs)
            labs = sorted(glob.glob(join(path, 'GT_full', sample, '*.png')))
            self.lab_paths.extend(labs)

class DatasetInfer3D(Dataset):
    def __init__(self, args, mode='test'):
        self.args = args
        self.mode = mode
        data_path = join(args.infer_root, args.InferSet)
        self._set_file_paths(data_path, args.split_path)
        
        print('All data load done, length of images:', len(self.img_paths))
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        
        img_nii = sitk.ReadImage(self.img_paths[index])
        img = sitk.GetArrayFromImage(img_nii).astype(np.float32)
        lab_nii = sitk.ReadImage(self.lab_paths[index])
        lab = sitk.GetArrayFromImage(lab_nii).astype(np.int8)
        
        if self.args.dataset == 'Covid1920_1p5':
            img = np.clip(img, -1300, 300)
        elif self.args.dataset[:4] == 'LiTS':
            img = np.clip(img, -17, 200)
                
        mean = np.mean(img)
        std = np.std(img)

        img -= mean
        img /= std
        
        # z, y, x = img.shape
        # Z, Y, X = self.args.training_size
        # zdiff, ydiff, xdiff = 0, 0, 0
        # # pad if the image size is smaller than trainig size
        # if z < Z:
        #     zdiff = int(math.ceil((Z - z) / 2)) 
        #     img = np.pad(img, ((zdiff, Z-z-zdiff), (0,0), (0,0)))
        # if y < Y:
        #     ydiff = int(math.ceil((Y - y) / 2)) 
        #     img = np.pad(img, ((0,0), (ydiff, Y-y-ydiff), (0,0)))
        # if x < X:
        #     xdiff = int(math.ceil((X - x) / 2)) 
        #     img = np.pad(img, ((0,0), (0,0), (xdiff, X-x-xdiff)))
        
        # ori_size = torch.Tensor([z, y, x]).to(torch.int16)
        # pad_para = torch.Tensor([zdiff, ydiff, xdiff]).to(torch.int16)
        
        tensor_img = torch.from_numpy(img).unsqueeze(0).float()
        tensor_lab = torch.from_numpy(lab).unsqueeze(0).to(torch.uint8)

        casename = os.path.basename(self.img_paths[index]).split('.nii.gz')[0]

        meta_info = {"casename": casename,
                    #  "ori_size": ori_size,
                    #  "pad_para": pad_para,
                     "origin": torch.Tensor(img_nii.GetOrigin()).to(torch.float),
                     "direction": torch.Tensor(img_nii.GetDirection()).to(torch.float),
                     "spacing": torch.Tensor(img_nii.GetSpacing()).to(torch.float)
                     }
        
        return (tensor_img.float(), tensor_lab.to(torch.int8), meta_info)

    def _set_file_paths(self, path, split_path):
        self.img_paths, self.lab_paths = [], []
        split_info = json.load(open(split_path))

        for sample in split_info[self.mode]:
            self.img_paths.append(join(path, 'IMG', sample+'.nii.gz'))
            self.lab_paths.append(join(path, 'GT_full', sample+'_full.nii.gz'))

class DatasetInfer3D_ISELE(Dataset):
    def __init__(self, args, mode='test'):
        self.args = args
        self.mode = mode
        data_path = join(args.infer_root, args.InferSet)
        self._set_file_paths(data_path, args.split_path)
        
        print('All data load done, length of images:', len(self.imgadc_paths))
    
    def __len__(self):
        return len(self.imgadc_paths)
    
    def __getitem__(self, index):
        
        img_adcnii = sitk.ReadImage(self.imgadc_paths[index])
        img_adc = sitk.GetArrayFromImage(img_adcnii).astype(np.float32)
        img_dwinii = sitk.ReadImage(self.imgdwi_paths[index])
        img_dwi = sitk.GetArrayFromImage(img_dwinii).astype(np.float32)
        lab_nii = sitk.ReadImage(self.lab_paths[index])
        lab = sitk.GetArrayFromImage(lab_nii).astype(np.int8)
        
        percentile_2 = np.percentile(img_adc, 2, axis=None)
        percentile_98 = np.percentile(img_adc, 98, axis=None)
        adcimg = np.clip(img_adc, percentile_2, percentile_98)
        mean = np.mean(adcimg)
        std = np.std(adcimg)

        adcimg -= mean
        adcimg /= std
        
        percentile_2 = np.percentile(img_dwi, 2, axis=None)
        percentile_98 = np.percentile(img_dwi, 98, axis=None)
        dwiimg = np.clip(img_dwi, percentile_2, percentile_98)
        
        mean = np.mean(dwiimg)
        std = np.std(dwiimg)

        dwiimg -= mean
        dwiimg /= std
        
        img = np.stack((adcimg, dwiimg), axis=0)
        tensor_img = torch.from_numpy(img).float()
        tensor_lab = torch.from_numpy(lab).unsqueeze(0).to(torch.uint8)

        casename = os.path.basename(self.imgadc_paths[index]).split('_adc.nii.gz')[0]

        meta_info = {"casename": casename,
                     "origin": torch.Tensor(img_adcnii.GetOrigin()).to(torch.float),
                     "direction": torch.Tensor(img_adcnii.GetDirection()).to(torch.float),
                     "spacing": torch.Tensor(img_adcnii.GetSpacing()).to(torch.float)
                     }
        
        return (tensor_img.float(), tensor_lab.to(torch.int8), meta_info)

    def _set_file_paths(self, path, split_path):
        self.imgadc_paths, self.imgdwi_paths, self.lab_paths = [], [], []
        split_info = json.load(open(split_path))

        for sample in split_info[self.mode]:
            self.imgadc_paths.append(join(path, 'IMG_adc', sample+'_adc.nii.gz'))
            self.imgdwi_paths.append(join(path, 'IMG_dwi', sample+'_dwi.nii.gz'))
            self.lab_paths.append(join(path, 'GT_full', sample+'_full.nii.gz'))

class DatasetInfer3D_MS(Dataset):
    def __init__(self, args, mode='test'):
        self.args = args
        self.mode = mode
        data_path = join(args.infer_root, args.InferSet)
        self._set_file_paths(data_path, args.split_path)
        
        print('All data load done, length of images:', len(self.img_paths))
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        
        img_nii = sitk.ReadImage(self.img_paths[index])
        img = sitk.GetArrayFromImage(img_nii).astype(np.float32)
        lab_nii = sitk.ReadImage(self.lab_paths[index])
        lab = sitk.GetArrayFromImage(lab_nii).astype(np.int8)
        
        percentile_2 = np.percentile(img, 2, axis=None)
        percentile_98 = np.percentile(img, 98, axis=None)
        img = np.clip(img, percentile_2, percentile_98)
        mean = np.mean(img)
        std = np.std(img)

        img -= mean
        img /= std
        
        tensor_img = torch.from_numpy(img).unsqueeze(0).float()
        tensor_lab = torch.from_numpy(lab).unsqueeze(0).to(torch.uint8)

        casename = os.path.basename(self.img_paths[index]).split('.nii.gz')[0]

        meta_info = {"casename": casename,
                     "origin": torch.Tensor(img_nii.GetOrigin()).to(torch.float),
                     "direction": torch.Tensor(img_nii.GetDirection()).to(torch.float),
                     "spacing": torch.Tensor(img_nii.GetSpacing()).to(torch.float)
                     }
        
        return (tensor_img.float(), tensor_lab.to(torch.int8), meta_info)

    def _set_file_paths(self, path, split_path):
        self.img_paths, self.lab_paths = [], []
        split_info = json.load(open(split_path))

        for sample in split_info[self.mode]:
            self.img_paths.append(join(path, 'IMG', sample+'.nii.gz'))
            self.lab_paths.append(join(path, 'GT_full', sample+'_full.nii.gz'))


if __name__ == '__main__':
    from torch.utils import data
    
    class Args(object):
        def __init__(self):
            self.data_root = '/mnt/data/gxy/PIA_Data/npy/2D'
            self.dataset = 'Covid1920'
            self.lab_type = 'part'
            self.color_prob = 0.3
            self.flip_prob = 0.3
            self.batch_size = 4
            self.num_workers = 2
    
    args = Args()
    
    trainset, suppset = Dataset2D(args, mode='train'), Dataset2D(args, mode='supp')
    valset, testset = Dataset2D(args, mode='val'), Dataset2D(args, mode='test')
    
    suppLoader = data.DataLoader(suppset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    