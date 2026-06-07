from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, requests, io, sys, time
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
Q_batch_size = 1
assert Q_batch_size == 1

model_path = "/data2/Qwen/Qwen2.5-7B"
beta = 0.04
num_pre_Q = 8
all_steps = 1000
max_prompt_length = 400   
save_steps = 200
compute_gen_logps = True
clip_param = 0.2
ref_server = "http://localhost:59875"
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list