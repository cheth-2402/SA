from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import os
import numpy as np
import sys
import re
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
import random
import time
import torch.distributed as dist
from mmcv.runner import get_dist_info
from transformers import T5EncoderModel, T5Tokenizer
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs
from mmcv.runner import LogBuffer
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

from src.editor.model import SlotEditor
from train_scripts.train import image_from_tensor, create_colored_combined_mask, color_map
from src.utils.misc import read_config
from src.slot_attention import UOD

from diffusers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

import datetime
import pickle
import yaml, json
from PIL import Image



@dataclass
class TrainingConfig:
    #Model
    hidden_size:int = 512
    depth:int = 4
    num_heads:int = 8
    mlp_ratio:float = 4.0
    model_max_length:int = 300
    caption_channels:int = 4096
    slot_dim:int = 64


    #Data
    train_pickle_file:str = "/home/scai/phd/aiz228170/scratch/PixArt-sigma/Editor/val_data_latest.pickle"
    val_pickle_file:str = "/home/scai/phd/aiz228170/scratch/PixArt-sigma/Editor/val_data_latest.pickle"
    batch_size:int = 64
    use_saved_t5:bool = True
    num_workers=1

    #Training
    mixed_precision:str = 'fp16'
    report_to:str = 'wandb'
    tracker_project_name:str = 'slot_editor'
    work_dir:str = './editor_runs/512_4_8_4_300_4096_64_64_1_1e-4_1000_on_val'
    gradient_accumulation_steps:int = 1
    use_fsdp: bool = False
    lr:float =1e-4
    num_warmup_steps:int =1000
    log_interval:int = 20
    seed:int = 0
    num_epochs:int = 100
    resume_from:str = None
    gradient_clip:float = 10.0
    save_model_steps:int = 5000
    save_model_epochs:int = 10
    pipeline_load_from:str = '/home/scai/phd/aiz228170/scratch/PixArt-sigma/output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers'

    #Visualization
    visualize:bool =True
    eval_sampling_steps:int = 1000
    samples_2_show: int = 8
    image_root:str = '/home/scai/phd/aiz228170/scratch/Datasets/CIM-NLI/combined/valid' #change this if using val data for val
    config_file:str = 'configs/our_config.py'
    checkpoint_path: str = '/scratch/cse/btech/cs1210561/SA/output/clevr_run_res112/checkpoints/epoch_468_step_524999.pth'




#SAVE and LOAD_CHECKPOINT
def save_checkpoint(work_dir,
                    epoch,
                    model,
                    model_ema=None,
                    optimizer=None,
                    lr_scheduler=None,
                    keep_last=False,
                    step=None):
    os.makedirs(work_dir, exist_ok=True)
    state_dict = dict(state_dict=model.state_dict())
    if model_ema is not None:
        state_dict['state_dict_ema'] = model_ema.state_dict()
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        state_dict['scheduler'] = lr_scheduler.state_dict()
    if epoch is not None:
        state_dict['epoch'] = epoch
        file_path = os.path.join(work_dir, f"epoch_{epoch}.pth")
        if step is not None:
            file_path = file_path.split('.pth')[0] + f"_step_{step}.pth"
    

    torch.save(state_dict, file_path)
    print(f'Saved checkpoint of epoch {epoch} to {file_path.format(epoch)}.')
    if keep_last:
        for i in range(epoch):
            previous_ckgt = file_path.format(i)
            if os.path.exists(previous_ckgt):
                os.remove(previous_ckgt)



def load_checkpoint(checkpoint,
                    model,
                    model_ema=None,
                    optimizer=None,
                    lr_scheduler=None,
                    load_ema=False,
                    resume_optimizer=True,
                    resume_lr_scheduler=True,
                    max_length=300,
                    ):
    assert isinstance(checkpoint, str)
    ckpt_file = checkpoint
    checkpoint = torch.load(ckpt_file, map_location="cpu")
    if load_ema:
        state_dict = checkpoint['state_dict_ema']
    else:
        state_dict = checkpoint.get('state_dict', checkpoint)

    missing, unexpect = model.load_state_dict(state_dict, strict=False)
    if model_ema is not None:
        model_ema.load_state_dict(checkpoint['state_dict_ema'], strict=False)
    if optimizer is not None and resume_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None and resume_lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint['scheduler'])

    if optimizer is not None:
        epoch = checkpoint.get('epoch', re.match(r'.*epoch_(\d*).*.pth', ckpt_file).group()[0])
        print(f'Resume checkpoint of epoch {epoch} from {ckpt_file}. Load ema: {load_ema}, '
                    f'resume optimizer： {resume_optimizer}, resume lr scheduler: {resume_lr_scheduler}.')
        return epoch, missing, unexpect
    print(f'Load checkpoint from {ckpt_file}. Load ema: {load_ema}.')
    return missing, unexpect



def init_random_seed(seed=None, device="cuda"):
    if seed is not None:
        return seed

    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()



def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def cosine_hungarian_matching_loss(outputs, targets, temperature=1.0):
    '''
    outputs: Batch x num_slots x slot_dim
    targets: Batch x num_slots x slot_dim
    '''
    row_i = []
    col_i = []
    bs, num_queries = outputs.shape[:2]

    # (Batch * num_slots x slot_dim)
    out_features = outputs.flatten(0, 1)
    tgt_features = targets.flatten(0, 1)

    # Normalize features
    out_features = F.normalize(out_features, p=2, dim=1)
    tgt_features = F.normalize(tgt_features, p=2, dim=1)

    # Cosine Similarity in one-go
    cost_matrix = -(torch.mm(out_features, tgt_features.t())  /temperature)

    C = cost_matrix.view(bs, num_queries, bs, num_queries)
    C_np = C.clone().cpu()

    with torch.no_grad():
        for b in range(bs):
            row_indices, col_indices = linear_sum_assignment(C_np[b,:,b,:])
            row_i.append(row_indices)
            col_i.append(col_indices)

    row_indices = torch.tensor(np.array(row_i), device=C.device)
    col_indices = torch.tensor(np.array(col_i), device=C.device)
    num_indices = row_indices.size(1)
    batch_indices = torch.arange(bs).unsqueeze(1).expand(-1,num_indices).to(C.device)
    values = C[batch_indices, row_indices, batch_indices, col_indices]
    # scaling b/w [0,1]
    loss_val = (values.mean() + 1.)/2
    return loss_val




class SlotDataset(Dataset):
    def __init__(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        self.data = data
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dct = self.data[index]

        return {
            "image_name": dct['inp_image_name'],
            "out_image_name": dct['out_image_name'],
            "edit_prompt": dct['edit_prompt'],
            "input_slots": dct['input_image_slots'],
            "output_slots": dct['output_image_slots'],
            "text_emb": dct['prompt_embedding'],
            'emb_mask': dct['embedding_mask']
        }
        


def unnormalize(images):
    std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
 
    images = images*std.to(images.device)
    images = images+mean.to(images.device)
    return images



@torch.inference_mode()
def log_validation(model, cfg, val_batch, global_step, device, samples_2_show=4, text_encoder=None, tokenizer=None):
    torch.cuda.empty_cache()
    model = accelerator.unwrap_model(model).eval()
    config = read_config(cfg.config_file)
    vis_model = UOD(config)

    checkpoint = torch.load(cfg.checkpoint_path)
    vis_model.load_state_dict(checkpoint['state_dict'])
    vis_model = vis_model.eval().to(device)


    input_slots = val_batch['input_slots'][:samples_2_show].to(device)
    target_slots = val_batch['output_slots'][:samples_2_show].to(device)

    if use_saved_t5 and (text_encoder is None and tokenizer is None):
        y = val_batch['text_emb'][:samples_2_show].to(device)
        y_mask = val_batch['emb_mask'][:samples_2_show].to(device)
    else:
        txt_tokens = tokenizer(val_batch['edit_prompt'][:samples_2_show], max_length=max_length, padding='max_length', truncation=True,
        return_tensors='pt').to(accelerator.device)
        
        y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:,None]
        y_mask = y_mask = txt_tokens.attention_mask[:, None, None]
    

    inp_image_names = val_batch['image_name'][:samples_2_show]
    out_image_names = val_batch['out_image_name'][:samples_2_show]


    src_images = [Image.open(os.path.join(val_image_root,'images', k)).convert('RGB').resize((224,224)) for k in inp_image_names]
    tgt_images = [Image.open(os.path.join(val_image_root,'images_c1',k)).convert('RGB').resize((224,224)) for k in out_image_names]

    img_stack = [np.hstack((np.asarray(og),np.asarray(tgt))) for og,tgt in zip(src_images, tgt_images)]

    # print(len(img_stack), len(inp_image_names), input_slots.shape, y.shape, y_mask.shape)

    output_slots = model(input_slots, y, y_mask)
    # print(output_slots.shape)

    torch.cuda.empty_cache()

    dec_output = vis_model.slot_to_img(output_slots)
    gen_tensor = unnormalize(dec_output['generated'])
    
    bs = output_slots.shape[0]
    num_slots = output_slots.shape[1]

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            import wandb
            formatted_images = []
            for idx in range(bs):
                istack = img_stack[idx]
                image_reconstructed = image_from_tensor(gen_tensor[idx])
                slot_masks = np.hstack([Image.fromarray((dec_output['masks'][idx][k].squeeze().cpu().detach().numpy()*255).astype(np.uint8)).convert('RGB') for k in range(num_slots)])
                final_img = np.hstack((istack, image_reconstructed, slot_masks))
                image = wandb.Image(final_img, caption=val_batch['edit_prompt'][idx])
                formatted_images.append(image)
            tracker.log({"validation": formatted_images})
    
    del vis_model
    # return image_logs





        

def train():
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()
    global_step = start_step + 1

    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start = time.time()
        data_time_all = 0
        for step, batch in enumerate(train_dl):
            global_step += 1
            with torch.cuda.amp.autocast(enabled=(config.mixed_precision == 'fp16' or config.mixed_precision == 'bf16')):
                input_slots = batch['input_slots']
                target_slots = batch['output_slots']
        
            if use_saved_t5:
                y = batch['text_emb']
                y_mask = batch['emb_mask']
                tokenizer = None
                text_encoder = None
            else:
                with torch.no_grad():
                    txt_tokens = tokenizer(batch['edit_prompt'], max_length=max_length, padding='max_length', truncation=True,
                    return_tensors='pt').to(accelerator.device)
                    y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:,None]
                    y_mask = y_mask = txt_tokens.attention_mask[:, None, None]
        
            grad_norm = None
            data_time_all += time.time() - data_time_start
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                output_slots = model(input_slots, y, y_mask)
                # print(input_slots.shape, output_slots.shape, target_slots.shape)
                loss_term = cosine_hungarian_matching_loss(output_slots, target_slots)
                accelerator.backward(loss_term)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                lr_scheduler.step()
        
            lr = lr_scheduler.get_last_lr()[0]
            logs = {
                "loss": accelerator.gather(loss_term).mean().item(),
            }
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)

            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step + 1)
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dl) - step - 1))))
                log_buffer.average()
        
                info = f"Step/Epoch [{global_step}/{epoch}][{step+1}/{len(train_dl)}]: epoch_eta:{eta_epoch} time_all:{t:.3f},time_data:{t_d:.3f}, lr:{lr:.3e} "


                info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                print(info)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
            logs.update(lr=lr)
            accelerator.log(logs, step=global_step)
            global_step += 1
            data_time_start = time.time()


            if config.save_model_steps and global_step % config.save_model_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                        epoch=epoch,
                                        step=global_step,
                                        model=accelerator.unwrap_model(model),
                                        optimizer=optimizer,
                                        lr_scheduler=lr_scheduler
                                        )
            
            if config.visualize and (global_step % config.eval_sampling_steps == 0 or (step+1) == 1):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    log_validation(model, config, val_batch, global_step, device=accelerator.device, samples_2_show=4, text_encoder=text_encoder, tokenizer=tokenizer)

        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.umask(0o000)
                save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                epoch=epoch,
                                step=global_step,
                                model=accelerator.unwrap_model(model),
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler
                                )
        
        
        accelerator.wait_for_everyone()
        





            
    



if __name__ == '__main__':
    config = TrainingConfig()

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400) 


    if config.use_fsdp:
        raise Exception("Setup FSDP boi!!")
    else:
        init_train = 'DDP'
        fsdp_plugin = None
    
    even_batches = True
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=config.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler],
    )

    config.seed = init_random_seed(config.seed)
    set_random_seed(config.seed)
 

    if accelerator.is_main_process:
        cfg_dct = asdict(config)
        with open(os.path.join(config.work_dir, 'config.json'),'w') as f:
            json.dump(cfg_dct, f, indent=4)

    model = SlotEditor(
                hidden_size=config.hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                model_max_length=config.model_max_length,
                caption_channels=config.caption_channels,
                slot_dim=config.slot_dim
            )


    if not config.use_saved_t5:
        tokenizer = T5Tokenizer.from_pretrained(config.pipeline_load_from, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            config.pipeline_load_from, subfolder="text_encoder", torch_dtype=torch.float16).to(accelerator.device)

    if accelerator.is_main_process:
        print(f"Model Params : {sum(p.numel() for p in model.parameters()):}")
        print(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):}")


    train_ds = SlotDataset(config.train_pickle_file)
    val_ds = SlotDataset(config.train_pickle_file)

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=1)

    val_batch = next(iter(val_dl))


    if hasattr(model, 'module'):
        model = model.module
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)


    lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.num_warmup_steps
        )

    
    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        tracker_config['report_to'] = 'wandb'
        # print(tracker_config)
        try:
            accelerator.init_trackers(config.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")
    

    start_epoch = 0
    start_step = 0
    total_steps = len(train_dl) * config.num_epochs
    use_saved_t5 = config.use_saved_t5
    max_length = config.model_max_length
    val_image_root = config.image_root


    if config.resume_from is not None:
        _, missing, unexpected = load_checkpoint(config.resume_from,
                                                model=model,
                                                lr_scheduler=lr_scheduler,
                                                optimizer=optimizer,
                                                max_length=config.model_max_length)
        print(f'Missing keys: {missing}')
        print(f'Unexpected keys: {unexpected}')


    model = accelerator.prepare(model)
    optimizer, train_dl, val_dl, lr_scheduler = accelerator.prepare(optimizer, train_dl, val_dl, lr_scheduler)
    train()




# python -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 train_scripts/train_editor.py 