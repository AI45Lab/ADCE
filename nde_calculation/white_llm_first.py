import re
import torch
from tqdm import tqdm
import logging
import datetime
import argparse
import numpy as np
from colorama import Fore, Style
import os
import warnings
import json
import sys
import os
sys.path.append('../')
sys.path.append('./')
from utils.load_model import *
from utils.lm_hf import *
from utils.util import *
from utils.load_data import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Load and process datasets.")
parser.add_argument('--dataset_name', type=str, default='analytic_entailment', help='Name of the dataset to load')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
parser.add_argument('--shuffle', type=bool, default=False, help='Whether to shuffle the dataset')
parser.add_argument('--num_workers', type=int, default=4, help='The number of loading workers')
parser.add_argument('--model_name', type=str, default='llama-3-8b', help='Name of the model to load')
parser.add_argument('--gpu', type=int, default= 0, help='Device to use (e.g., cuda:0, cuda:1, cpu)')
parser.add_argument('--max_new_tokens', type=int, default= 20,help='The number of max generated token')
parser.add_argument('--prompt', type=str, default= 'superhigh', help='The level of prompts')
parser.add_argument('--folder_path', type=str, help='The folder path',default=None)
args = parser.parse_args()

# print(args)

load_model_config = load_models_config()
model_config_path =load_model_config.get(args.model_name, None)
with open(model_config_path+'/config.json', 'r') as f:
       args.model_config = json.load(f)
# args.log_file_path = f'{args.folder_path}running.log'


"""
Load Model and Data
"""
model, tokenizer = load_model(args.model_name, args.gpu)
print('Load model before finetuning.')

data_loader = load_dataloader(args, tokenizer = tokenizer)

"""
Evaluation and Extract activations
"""
if args.folder_path is None:
    now = datetime.datetime.now()
    current_time = now.strftime("%m%d_%H%M")
    args.folder_path =  f"./results/{args.dataset_name}/results_{args.prompt[0]}/acc_{args.model_name}_{current_time}"
    os.makedirs(args.folder_path, exist_ok=True)
logger = create_and_get_logger(args.folder_path)
llm = llm(model, tokenizer,data_loader,args, logger,args.folder_path)
print(Fore.GREEN + Style.BRIGHT + "Evaluation of LLM" + Style.RESET_ALL) 
max_input_ids = llm.evaluate_accuracy()
args.max_length = max_input_ids
logger.info(f"max length is \n  {args.max_length}\n")
print(Fore.GREEN + Style.BRIGHT + f"Activations of LLM with max sequence length {args.max_length}" + Style.RESET_ALL) 