from diffusers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math

from src.utils.logger import get_root_logger

from math import cos, pi 
import pdb
import numpy as np


class CosineAnnealWarmRestarts:

    def __init__(self, 
                 optimizer = None,
                 max_lr = 1e-4,
                 min_lr = 1e-6,
                 warmup_steps = 100, 
                 cycle_freq = 100,
                 cycle_start = -1,
                 cycle_freq_mult = 1,
                 steps = 0,
                 stay_low = True
                 ):
    
        assert optimizer
        self.optimizer = optimizer

        #the lrs
        num_grps = len(self.optimizer.param_groups)

        #set the lrs
        if(type(max_lr) == float):
            self.max_lr = [max_lr for i in range(num_grps)]
        else:
            assert len(max_lr) == num_grps
        
        if(type(min_lr) == float):
            self.min_lr = [min_lr for i in range(num_grps)]
        else:
            assert len(min_lr) == num_grps

        #current ti
        self.warmup_steps = warmup_steps
        self.delta = [(max_lr_i - min_lr_i) / self.warmup_steps for max_lr_i, min_lr_i in zip(self.max_lr, self.min_lr)]
        self.delta_2 = [(max_lr_i - min_lr_i)/2 for max_lr_i, min_lr_i in zip(self.max_lr, self.min_lr)]
        self.t_i = cycle_freq
        self.cycle_start = cycle_start
        self.steps = int(steps)
        self.t_curr = max(self.steps - self.cycle_start, 0)
        self.t_mult = cycle_freq_mult
        self.init_lr()
        self.stay_low = stay_low
        print(self.steps, self.t_curr, self.t_i)

    def init_lr(self):
        self.step()

    def set_lr(self):
        for lr, param in zip(self.lr, self.optimizer.param_groups):
            param['lr'] = lr
    
    def load_state_dict(self, ckpt):
        pass 
    
    def state_dict(self):
        pass 

    def get_last_lr(self):
        return self.lr

    def step(self):

        #are you in linear wamup?
        if(self.steps <= self.warmup_steps):
            self.lr = [lr_i + delta_i*self.steps for lr_i,delta_i in zip(self.min_lr, self.delta)]
        
        elif(self.steps <= self.cycle_start):
            self.lr = self.max_lr
        else:

            if(self.t_curr == self.t_i + 1):
                if(not self.stay_low):
                    self.t_curr = 0
                    self.t_i *= self.t_mult
                    self.t_curr += 1
                    cos_term = 1 + cos(self.t_curr/self.t_i*pi)
                    self.lr = [min_lr_i + delta_2_i*cos_term for min_lr_i, delta_2_i in zip(self.min_lr, self.delta_2)]
                
                else:
                    self.lr = self.min_lr
            else:
                cos_term = 1 + cos(self.t_curr/self.t_i*pi)
                self.lr = [min_lr_i + delta_2_i*cos_term for min_lr_i, delta_2_i in zip(self.min_lr, self.delta_2)]
                self.t_curr += 1
            
        self.set_lr()
        self.steps += 1



class ConstantWithWarmup:

    def __init__(self, 
                 optimizer = None,
                 max_lr = 1e-4,
                 min_lr = 1e-6,
                 warmup_steps = 100,
                 steps = 0 
                 ):
    
        assert optimizer
        self.optimizer = optimizer

        #the lrs
        num_grps = len(self.optimizer.param_groups)

        #set the lrs
        if(type(max_lr) == float):
            self.max_lr = [max_lr for i in range(num_grps)]
        else:
            assert len(max_lr) == num_grps
        
        if(type(min_lr) == float):
            self.min_lr = [min_lr for i in range(num_grps)]
        else:
            assert len(min_lr) == num_grps

        #current ti
        self.warmup_steps = warmup_steps
        self.delta = [(max_lr_i - min_lr_i) / self.warmup_steps for max_lr_i, min_lr_i in zip(self.max_lr, self.min_lr)]
        self.steps = int(steps)
        self.init_lr()
        
    def init_lr(self):
        self.step()

    def set_lr(self):
        for lr, param in zip(self.lr, self.optimizer.param_groups):
            param['lr'] = lr
    
    def load_state_dict(self, ckpt):
        pass 
    
    def state_dict(self):
        pass 

    def get_last_lr(self):
        return self.lr

    def step(self):

        #are you in linear wamup?
        if(self.steps <= self.warmup_steps):
            self.lr = [lr_i + delta_i*self.steps for lr_i,delta_i in zip(self.min_lr, self.delta)]
        
        else:
            self.lr = self.max_lr
        self.set_lr()
        self.steps += 1

def build_lr_scheduler_our(config, optimizer, train_dataloader, lr_scale_ratio):
    lr = config.optimizer.lr
    warmup_steps = config.lr_schedule_args.num_warmup_steps
    scheduler = ConstantWithWarmup(
        optimizer,
        lr,
        lr/100,
        warmup_steps
    )
    # lr = config.optimizer.lr 
    # total_steps = config.num_epochs*(len(train_dataloader))/config.gradient_accumulation_steps
    # warmup_steps = config.lr_schedule_args.num_warmup_steps
    # cycle_start = warmup_steps
    # cycle_length = total_steps - cycle_start
    # scheduler = CosineAnnealWarmRestarts(
    #     optimizer,
    #     lr,
    #     lr/100,
    #     warmup_steps = warmup_steps,
    #     cycle_freq = cycle_length,
    #     cycle_start = cycle_start,
    #     steps = 0
    # )
    return scheduler


def build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio):
    if not config.get('lr_schedule_args', None):
        config.lr_schedule_args = dict()
    if config.get('lr_warmup_steps', None):
        config['num_warmup_steps'] = config.get('lr_warmup_steps')  # for compatibility with old version

    logger = get_root_logger()
    logger.info(
        f'Lr schedule: {config.lr_schedule}, ' + ",".join(
            [f"{key}:{value}" for key, value in config.lr_schedule_args.items()]) + '.')
    if config.lr_schedule == 'cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            **config.lr_schedule_args,
            num_training_steps=(len(train_dataloader) * config.num_epochs),
        )
    elif config.lr_schedule == 'constant':
        lr_scheduler = get_constant_schedule(
            optimizer=optimizer,
        )
    elif config.lr_schedule == 'cosine_decay_to_constant':
        assert lr_scale_ratio >= 1
        lr_scheduler = get_cosine_decay_to_constant_with_warmup(
            optimizer=optimizer,
            **config.lr_schedule_args,
            final_lr=1 / lr_scale_ratio,
            num_training_steps=(len(train_dataloader) * config.num_epochs),
        )
    else:
        raise RuntimeError(f'Unrecognized lr schedule {config.lr_schedule}.')
    return lr_scheduler

def get_constant_schedule(optimizer: Optimizer, last_epoch: int = -1) -> LambdaLR:
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)

def get_cosine_decay_to_constant_with_warmup(optimizer: Optimizer,
                                             num_warmup_steps: int,
                                             num_training_steps: int,
                                             final_lr: float = 0.0,
                                             num_decay: float = 0.667,
                                             num_cycles: float = 0.5,
                                             last_epoch: int = -1
                                             ):
    """
    Create a schedule with a cosine annealing lr followed by a constant lr.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The number of total training steps.
        final_lr (`int`):
            The final constant lr after cosine decay.
        num_decay (`int`):
            The
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        num_decay_steps = int(num_training_steps * num_decay)
        if current_step > num_decay_steps:
            return final_lr

        progress = float(current_step - num_warmup_steps) / float(max(1, num_decay_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * (
                1 - final_lr) + final_lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)
