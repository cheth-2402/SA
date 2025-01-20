import h5py 
from tqdm import tqdm

import sys
from pathlib import Path
from PIL import Image, ImageFile



current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))


from src.utils.misc import read_config
from src.slot_attention import UOD
from src.data.builder import build_dataloader

from torch.utils.data import DataLoader, Dataset



class CLEVRTEXDataset(Dataset):
    def __init__(self, image_folder = "/home/cse/btech/cs1210561/scratch/CLEVRTEX_new/train.hdf5", image_size = 224):
        self.image_folder = image_folder
        self.train_images = h5py.File(image_folder, "r")
        self.train_images = self.train_images['images'][:]
        self.val_transform_image = transforms.Compose([
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)

    def __len__(self):
        return len(self.train_images)    
    
    def unormalize(self, images):
        images = images*self.std.to(images.device)
        images = images + self.mean.to(images.device)
        return images         

    def __getitem__(self, idx):
        image = self.train_images[idx]
        image = image.astype('float32') / 255.0
        image = torch.tensor(image).permute(2, 0, 1)
        image = self.val_transform_image(image)
        return image


class CLEVRTExDataset_Old(Dataset):
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
        return {'image':image, 'name':img_path.split('/')[-1]}

import os
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
dat = CLEVRTExDataset_Old(image_folder = "/home/cse/btech/cs1210561/scratch/CLEVRTEX_new/InternImgs")

# dnew = CLEVRTEXDataset()


config_file = '/home/cse/btech/cs1210561/scratch/SA/configs/our_config.py'
config = read_config(config_file)
model = UOD(config)

model = model.eval()
device = "cuda:0"

checkpoint_path = '/home/cse/btech/cs1210561/scratch/SA/output/clevr_run_res112/checkpoints/epoch_468_step_524999.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])
model = model.eval()
model = model.to(device)

train_dataloader = build_dataloader(dat, num_workers=config.num_workers, batch_size=config.train_batch_size, shuffle=True)
pbar = tqdm(enumerate(train_dataloader), ncols = 120)
pbar.set_postfix({"loss":100, "rank:": 1})

import pickle

all_results = []

step = 0

for i, batch in enumerate(pbar):
    model.eval()
    
    with torch.no_grad(): 
        images = batch[1]['image'].to(device)
        names = batch[1]['name']
        inp = {"images": images}

        results = model(inp)  
        slots = results['slots']

        print(f'steps: {step}', end = ' | ')
        print("loss: ", results['loss'].item()) 
        
        for name, slot in zip(names, slots):
            all_results.append({'name': name, 'slots': slot})  
        step+=1
    torch.cuda.empty_cache()

    
output_file = "image_slots_dump.pkl"
with open(output_file, "wb") as f:
    pickle.dump(all_results, f)

print(f"Data saved to {output_file}")
