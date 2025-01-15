import argparse
import datetime
import os
import sys
import time
import types
import warnings
from pathlib import Path
import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs
from accelerate.utils import DistributedType
from tqdm import tqdm
from mmcv.runner import LogBuffer
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile


current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))


from src.slot_attention import UOD
from src.data.builder import build_dataset, build_dataloader, set_data_root

from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_, flush
from src.utils.logger import get_root_logger, rename_file_with_creation_time
from src.utils.lr_scheduler import build_lr_scheduler, build_lr_scheduler_our
from src.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from src.utils.optimizer import build_optimizer, auto_scale_lr, build_optimizer_our

from evaluation.eval_slots_new import CLEVRTExDataset

import matplotlib.pyplot as plt 
def plot_lr(config, train_dataloader, scheduler_t):
    total_steps = config.num_epochs*(len(train_dataloader))/config.gradient_accumulation_steps
    lr = []
    x = []
    for i in tqdm(range(int(total_steps))):
        scheduler_t.step()
        lr.append(scheduler_t.get_last_lr()[0])
        x.append(i / len(train_dataloader)*config.gradient_accumulation_steps)
    #pdb.set_trace()
    plt.plot(x,lr)
    plt.savefig(os.path.join(config.work_dir, 'lr.png'))

def compute_grad_norm(model, filter_by_name=None, grad_scale=None):
    total_norm = 0
    parameters = [(n,p) for n,p in model.named_parameters() if (filter_by_name is None or filter_by_name in n) and p.grad is not None and p.requires_grad]
    for n,p in parameters:
        unscaled_grad = p.grad.detach().data / grad_scale
        param_norm = unscaled_grad.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def count_parameters(model):
    '''
    taken from
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/3
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

color_map = torch.tensor([
    [1., 0, 0],     # Object 1 (Red)
    [0., 1, 0],     # Object 2 (Green)
    [0, 0., 1],     # Object 3 (Blue)
    [1, 1., 0],   # Object 4 (Yellow)
    [1, 0., 1],   # Object 5 (Magenta)
    [0, 1, 1.],   # Object 6 (Cyan)
    [0.5, 0.5, 0.5], # Object 7 (Gray)
    [1, 0.5, 0],   # Object 8 (Orange)
    [0.5, 0, 0.5],   # Object 9 (Purple)
])

@torch.inference_mode()
def image_from_tensor(image):
    image = image.permute(1,2,0)
    # image = (image + 1)/2
    image = image*255
    image = image.cpu().detach().numpy()
    image = image.astype('uint8')
    image = Image.fromarray(image)
    return image

def create_colored_combined_mask(combined_mask):

    h, w = combined_mask.shape
    num_slots = len(color_map)
    
    colored_mask = torch.zeros((h, w, 3), device=combined_mask.device)
    
    for slot_idx in range(num_slots):
        slot_mask = (combined_mask == slot_idx).unsqueeze(-1).float()  # Shape: [H, W, 1]

        color_map_cuda = color_map.to(combined_mask.device)
        
        colored_mask += slot_mask * color_map_cuda[slot_idx].view(1, 1, 3)  # Broadcasting for RGB channels
    colored_mask = colored_mask*255
    colored_mask = colored_mask.cpu().detach().numpy()
    colored_mask = colored_mask.astype('uint8')
    colored_mask = Image.fromarray(colored_mask)
    return colored_mask


def log_validation(model, step, device):
    with torch.no_grad():
        torch.cuda.empty_cache()
        model = accelerator.unwrap_model(model).eval()

        val_dataset = CLEVRTExDataset(config.val_dataset_path,image_size = 224)
        val_loader = DataLoader(val_dataset, sampler=None, shuffle=False, drop_last = False, batch_size=32, num_workers=config.num_workers)
        results = {}

        with torch.no_grad():
            model.eval()

            val_mse = 0.
            counter = 0

            for batch, image in enumerate(tqdm(val_loader)):
                image = image.to(device)
                inp = {"images":image}
                results = model(inp)
                break
        orig_images = val_dataset.unormalize(results['images'])
        recon_images = val_dataset.unormalize(results['generated'])
        slot_masks = results['masks']
        combined_masks = slot_masks.argmax(1)[:,0]

        fig, axes = plt.subplots(batch_size, 3 + config.slot["number_of_slots"], figsize=(15, 5 * batch_size))

        for idx in range(batch_size):
            current_row = axes[idx] if batch_size > 1 else axes  
            current_row[0].imshow(image_from_tensor(orig_images[idx]))
            current_row[0].axis('off')
            current_row[0].set_title('Original')
            current_row[1].imshow(image_from_tensor(recon_images[idx]))
            current_row[1].axis('off')
            current_row[1].set_title('Reconstructed')

            comb_mask = create_colored_combined_mask(combined_masks[idx])
            current_row[2].imshow(comb_mask)
            current_row[2].axis('off')
            current_row[2].set_title('Combined Mask')

            for i in range(config.slot["number_of_slots"]):
                current_row[3 + i].imshow(slot_masks[idx][i].squeeze().cpu().detach().numpy(), cmap='gray')
                current_row[3 + i].axis('off')
                current_row[3 + i].set_title(f'Slot {i+1}')

        plt.tight_layout()
        plt.show()
        folder = os.path.join(config.work_dir, f'vis_{step}')
        os.makedirs(folder, exist_ok = True)
        fig_path = os.path.join(folder, f"batch_{step}.png")
        fig.savefig(fig_path)

        import wandb
        wandb_images = {
            'batch_image': wandb.Image(fig_path)
        }

        tensorboard_images = {
            'batch_image': np.array(fig.canvas.buffer_rgba())
        }
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tracker.writer.add_image('batch_image', tensorboard_images['batch_image'], step)
            elif tracker.name == "wandb":
                tracker.log(wandb_images)
            else:
                logger.warn(f"Image logging not implemented for {tracker.name}")
        flush()



def train(step):
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    #step = 0

    rank = accelerator.state.process_index
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start= time.time()
        data_time_all = 0

        pbar = tqdm(enumerate(train_dataloader), ncols = 120)
        pbar.set_postfix({"loss":100, "rank:": rank})

        #the logs
        logs = dict()
        logs['loss'] = []
        logs['grad_norm'] = []
            
        
        for i, batch in pbar:
            model.train()

            images = batch[0]
            inp = {"images":images}
            results = model(inp)
            loss = results['loss']
            

            accelerator.backward(loss  / config.gradient_accumulation_steps)
            logs['loss'].append(loss.item())

            grad_scale = accelerator.scaler.get_scale()
            
            grad_norm = compute_grad_norm(model, grad_scale=grad_scale)
            logs['grad_norm'].append(grad_norm)


            if ((i+1) % config.gradient_accumulation_steps == 0):

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                lr = lr_scheduler.get_last_lr()[0]
                

                pbar.set_postfix({"loss":loss.item(),  "rank:": rank})
                pbar.update()
            
                if step % config.log_interval == 0:
                    if(accelerator.is_main_process):
                        
                        logs['lr'] = [lr]
                        print(f'steps: {step}', end = ' | ')
                        for k in logs:
                            logs[k] = sum(logs[k]) / len(logs[k])
                            print(f"{k}: {logs[k]}", end=" | ")
                        print()
                        accelerator.log(logs, step = step)
                        for k in logs:
                            logs[k] = []

                if config.save_model_steps and (step + 1) % config.save_model_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        os.umask(0o000)
                        save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                        epoch=epoch,
                                        step=step,
                                        model=accelerator.unwrap_model(model),
                                        optimizer=optimizer,
                                        lr_scheduler=lr_scheduler
                                        )
                if config.visualize and (step % config.eval_sampling_steps == 0 or (step + 1) == 1):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        log_validation(model, step, device=accelerator.device)

                if step % config.compute_metrics_every_n == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        # compute_slot_metrics(model, step, device=accelerator.device)
                        pass

                accelerator.wait_for_everyone()

                step += 1

        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.umask(0o000)
                save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                epoch=epoch,
                                step=step,
                                model=accelerator.unwrap_model(model),
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler
                                )
            


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument('--resume-from', help='the dir to resume the training')
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    if args.resume_from is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True)

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    
    init_train = 'DDP'
    fsdp_plugin = None
    even_batches = True

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler],
    )

    log_name = 'train_log.log'
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(config.work_dir, log_name)):
            rename_file_with_creation_time(os.path.join(config.work_dir, log_name))
    logger = get_root_logger(os.path.join(config.work_dir, log_name))

    logger.info(accelerator.state)
    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.image_size  # @param [256, 512]

    model = UOD(config)

    step = 0

    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"{model.__class__.__name__} Trainable Parameters: {count_parameters(model)}")

    set_data_root(config.data_root)

    #Dummy
    max_length = 300

    # dataset = build_dataset(
    #             config.data, resolution=image_size,
    #             max_length=max_length, config=config,
    #         )
    dataset = CLEVRTExDataset(config.train_dataset_path,image_size = 224)

    num_devices = accelerator.num_processes
    batch_size = config.train_batch_size // num_devices
    batch_size = batch_size // config.gradient_accumulation_steps
    print(f'Batch size: {batch_size}, effective batch size: {config.train_batch_size}')
    train_dataloader = build_dataloader(dataset, num_workers=config.num_workers, batch_size=batch_size, shuffle=True)

    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
                                       config.optimizer, **config.auto_lr)
    optimizer = build_optimizer_our(model, config.optimizer)

    #plot the lr 
    if accelerator.is_main_process:
        lr_scheduler = build_lr_scheduler_our(config, optimizer, train_dataloader, lr_scale_ratio)
        plot_lr(config, train_dataloader, lr_scheduler)
    lr_scheduler = build_lr_scheduler_our(config, optimizer, train_dataloader, lr_scale_ratio)

    if config.checkpoint_path is not None:
        checkpoint = torch.load(config.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        path = config.checkpoint_path.split('.')[-2]
        step = int(path.split('_')[-1])
        lr_scheduler.steps = step
        print(f"loaded checkpoint epoch: steps: {step}")

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(config.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    total_steps = len(train_dataloader) * config.num_epochs

    
    model = accelerator.prepare(model)
  
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
    train(step)