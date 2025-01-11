decay_rate = 0.5
decay_steps = 100000

"COnfirm this!!"
dataset = "BB_MMM_RGB"


data_root = '/home/cse/btech/cs1210561/scratch/CLEVRTEX_new'
train_dataset_path = '/home/cse/btech/cs1210561/scratch/CLEVRTEX_new/InternImgs'
val_dataset_path = '/home/cse/btech/cs1210561/scratch/CLEVRTEX_val/InternImgs'
image_list_json = ['data_info.json']

#Confirm
image_size = 224

#Using Single scale data
data = dict(
    type='InternalDataSigma', root='InternData', image_list_json=image_list_json, transform='default_train',
    load_vae_feat=False
)

device = "cuda:0"
decoder = dict(
    hidden_channels= 64,
    in_channels = 64,
    initial_resolution = (14,14),
    out_channels = 4
)
encoder = dict(
    hidden_channels= 64,
    in_channels = 3,
    out_channels = 64
)

hidden_dim = 64
learning_rate = 0.0004
model = 'slot_attention'

mixed_precision = 'fp16'  # ['fp16', 'fp32', 'bf16']
fp32_attention = True

work_dir = 'output/clevr_run1'


slot = dict(
  eps = 1.0e-07,
  gru = dict(
    hidden_dim = 64
  ),
  hidden_dim = 64,
  hidden_dimension_kv= 64,
  input_dim= 64,
  iterations= 3,
  number_of_slots= 7,
  slot_dim= 64
)

num_workers = 6
train_batch_size = 64 # 48 as default
eval_batch_size = 4
num_epochs = 100  # 3
gradient_accumulation_steps = 4
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='CAMEWrapper', lr=2e-5, weight_decay=0.0, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
lr_schedule_args = dict(num_warmup_steps=1000)
#lr_schedule = 'cosine_decay_to_constant'
lr_schedule = 'warmup_constant'

# train = dict(
#     batch_size= 64
#     data_path = ""
# )  
# val = dict(
#   batch_size= 32
#   data_path=""
#   sample_number= 30
# )
# validate_every=100
log_interval = 100
save_model_epochs = 5
save_model_steps = 2500
visualize = True
eval_sampling_steps = 100

compute_metrics_every_n = 2000
tracker_project_name="slot_diff_pixart"
checkpoint_path = None