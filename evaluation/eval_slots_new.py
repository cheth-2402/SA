import copy
import os
import os.path
import argparse
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(f"Path added = {os.path.dirname(os.path.abspath(__file__))}")
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils
from torchvision.utils import save_image
from ocl_metrics import UnsupervisedMaskIoUMetric, ARIMetric
from torchvision import transforms
import torchvision.transforms.functional as TF

from pycocotools import mask
from pycocotools.coco import COCO

from pathlib import Path
from PIL import Image, ImageFile
import numpy as np
import pdb

import sys
sys.path.insert(0, "/home/scai/phd/aiz228170/scratch/PixArt-sigma")
from diffusion.model.nets.slot.Img2SlotModel import ImageToSlotsModel
from diffusion.utils.misc import read_config

class SlotMetrics:
    def __init__(
        self,
        val_dataset_path="/home/cse/btech/cs1210561/scratch/CLEVRTEX_val/InternImgs",
        image_size=224,
        mask_size=320,
        num_workers=2,
        device="cuda"
    ):
        self.device = device
        val_dataset = CLEVRTExDataset(val_dataset_path,image_size = 224)
        max_tokens = int((320/16)**2)
        self.val_loader = DataLoader(val_dataset, sampler=None, shuffle=False, drop_last = False, batch_size=32, num_workers=num_workers)
        self.mask_size = mask_size

        MBO_c_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).to(device)
        MBO_i_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).to(device)
        miou_metric = UnsupervisedMaskIoUMetric(matching="hungarian", ignore_background = True, ignore_overlaps = True).to(device)
        ari_metric = ARIMetric(foreground = True, ignore_overlaps = True).to(device)
        
        MBO_c_slot_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).to(device)
        MBO_i_slot_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).to(device)
        miou_slot_metric = UnsupervisedMaskIoUMetric(matching="hungarian", ignore_background = True, ignore_overlaps = True).to(device)
        ari_slot_metric = ARIMetric(foreground = True, ignore_overlaps = True).to(device)


        self.METRICS = {
            "SLOT":
                {
                    "MBO_c": MBO_c_slot_metric,
                    "MBO_i": MBO_i_slot_metric,
                    "miou": miou_slot_metric,
                    "ari": ari_slot_metric
                },
            "SLOT-DECODER":
                {
                    "MBO_c": MBO_c_metric,
                    "MBO_i": MBO_i_metric,
                    "miou": miou_metric,
                    "ari": ari_metric
                }
        }

    def run_evaluation(self, model):
        with torch.no_grad():
            model.eval()

            val_mse = 0.
            counter = 0

            for batch, image in enumerate(tqdm(self.val_loader)):
                image = image.to(self.device)
                # true_mask_i = true_mask_i.to(self.device)
                # true_mask_c = true_mask_c.to(self.device)
                # mask_ignore = mask_ignore.to(self.device)
                
                batch_size = image.shape[0]
                counter += batch_size

                true_mask_i = torch.ones(batch_size,self.mask_size,self.mask_size).to(self.device).long()
                true_mask_c = torch.ones(batch_size,self.mask_size,self.mask_size).to(self.device).long()
                mask_ignore = torch.ones(batch_size,1,self.mask_size,self.mask_size).to(self.device).long()
                

                mse, default_slots_attns, dec_slots_attns, slots, dec_recon, attn_logits = model(image)

                default_attns = F.interpolate(default_slots_attns, size=320, mode='bilinear')
                dec_attns = F.interpolate(dec_slots_attns, size=320, mode='bilinear')
                
                default_attns = default_attns.unsqueeze(2)
                dec_attns = dec_attns.unsqueeze(2) # shape [B, num_slots, 1, H, W]


                pred_default_mask = default_attns.argmax(1).squeeze(1)
                pred_dec_mask = dec_attns.argmax(1).squeeze(1)

                val_mse += mse.item()

                true_mask_i_reshaped = torch.nn.functional.one_hot(true_mask_i).to(torch.float32).permute(0,3,1,2).to(self.device)
                true_mask_c_reshaped = torch.nn.functional.one_hot(true_mask_c).to(torch.float32).permute(0,3,1,2).to(self.device)
                pred_dec_mask_reshaped = torch.nn.functional.one_hot(pred_dec_mask).to(torch.float32).permute(0,3,1,2).to(self.device)
                pred_default_mask_reshaped = torch.nn.functional.one_hot(pred_default_mask).to(torch.float32).permute(0,3,1,2).to(self.device)

                self.METRICS["SLOT-DECODER"]["MBO_i"].update(pred_dec_mask_reshaped, true_mask_i_reshaped, mask_ignore)
                self.METRICS["SLOT-DECODER"]["MBO_c"].update(pred_dec_mask_reshaped, true_mask_c_reshaped, mask_ignore)
                self.METRICS["SLOT-DECODER"]["miou"].update(pred_dec_mask_reshaped, true_mask_i_reshaped, mask_ignore)
                self.METRICS["SLOT-DECODER"]["ari"].update(pred_dec_mask_reshaped, true_mask_i_reshaped, mask_ignore)


                self.METRICS["SLOT"]["MBO_i"].update(pred_default_mask_reshaped, true_mask_i_reshaped, mask_ignore)
                self.METRICS["SLOT"]["MBO_c"].update(pred_default_mask_reshaped, true_mask_c_reshaped, mask_ignore)
                self.METRICS["SLOT"]["miou"].update(pred_default_mask_reshaped, true_mask_i_reshaped, mask_ignore)
                self.METRICS["SLOT"]["ari"].update(pred_default_mask_reshaped, true_mask_i_reshaped, mask_ignore)


            scalar_metrics = {}
            for main_keys in self.METRICS:
                scalar_metrics[main_keys] = {}
                for k,v in self.METRICS[main_keys].items():
                    if 'ari' in k and (v.total == 0):
                        scalar_metrics[main_keys][k] = 0
                    else:
                        scalar_metrics[main_keys][k] = 100 * (v.values/v.total).item()
            
        return scalar_metrics



class CLEVRTExDataset(Dataset):
    def __init__(self, image_folder, image_size = 224):
        self.image_folder = image_folder
        self.image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))]
        self.val_transform_image = transforms.Compose([transforms.Resize(size = image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                               transforms.CenterCrop(size = image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)

    def __len__(self):
        return len(self.image_files)    
    
    def unormalize(self, images):

        images = images*self.std.to(images.device)
        images = images + self.mean.to(images.device)
        return images 


    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.val_transform_image(image)
        return image

class SlotVisualizer_new:
    def __init__(
        self,
        val_dataset_path="/home/cse/btech/cs1210561/scratch/CLEVRTEX_val/InternImgs",
        image_size=224,
        mask_size=320,
        num_workers=2,
        device="cuda"
    ):
        self.device = device
        val_dataset = CLEVRTExDataset(val_dataset_path,image_size = 224)
        max_tokens = int((320/16)**2)
        self.val_loader = DataLoader(val_dataset, sampler=None, shuffle=False, drop_last = False, batch_size=32, num_workers=num_workers)
        self.mask_size = mask_size


    def visualize(self, model):
        with torch.no_grad():
            model.eval()

            val_mse = 0.
            counter = 0

            for batch, image in enumerate(tqdm(self.val_loader)):
                image = image.to(self.device)

                return model(image)
                
                # batch_size = image.shape[0]
                # counter += batch_size

                # mse, default_slots_attns, dec_slots_attns, slots, dec_recon, attn_logits = model(image)

                # default_attns = F.interpolate(default_slots_attns, size=320, mode='bilinear')
                # dec_attns = F.interpolate(dec_slots_attns, size=320, mode='bilinear')
                
                # default_attns = default_attns.unsqueeze(2)
                # dec_attns = dec_attns.unsqueeze(2) # shape [B, num_slots, 1, H, W]

                # true_mask_i = torch.ones(batch_size,self.mask_size,self.mask_size).to(self.device).long()
                # true_mask_c = torch.ones(batch_size,self.mask_size,self.mask_size).to(self.device).long()

                # image = self.val_loader.dataset.unormalize(image)
                # results = {
                #     'images': image,
                #     'slots': slots,
                #     'decoder_masks': dec_attns,
                #     'encoder_masks': default_attns,
                #     'true_mask_instance': true_mask_i,
                #     'true_mask_class':true_mask_c
                # }
                # return results


if __name__ == "__main__":
    CKPT_PATH = '/home/scai/phd/aiz228170/scratch/PixArt-sigma/output/coco_spot_scratch_20_ep_bs_64/checkpoints/epoch_5_step_7500.pth'
    sd = torch.load(CKPT_PATH, 'cpu')
    CONFIG = read_config("/home/scai/phd/aiz228170/scratch/PixArt-sigma/output/coco_spot_scratch_20_ep_bs_64/config.py")
    kwargs = {'config': CONFIG}
    model = ImageToSlotsModel(**kwargs)
    SM_SD = {}
    for k,v in sd['state_dict'].items():
        if 'slot_model' in k:
            SM_SD[k.replace('slot_model.','')] = v

    model.load_state_dict(SM_SD)
    model = model.cuda()


    METRICS = SlotMetrics(device="cuda").run_evaluation(model)
    print(METRICS)