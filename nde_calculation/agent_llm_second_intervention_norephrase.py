import re
import torch
from tqdm import tqdm
import logging
import datetime
import argparse
import numpy as np
import json
from colorama import Fore, Style
import os
import warnings
from torch.utils.data import Dataset, DataLoader
import sys
import os
sys.path.append('../')
sys.path.append('./')
from utils.load_data import *
from utils.load_model import *
from utils.lm_hf import *
from utils.util import *
from utils.mask import MaskedWordDataset,collate_fn
from utils.agent_utils import llm_agent

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Load and process datasets.")

parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
parser.add_argument('--shuffle', type=bool, default=False, help='Whether to shuffle the dataset')
parser.add_argument('--num_workers', type=int, default=2, help='The number of loading workers')
parser.add_argument('--model_name', type=str, default='claude', help='Name of the model to load')
parser.add_argument('--version', type=str, default='claude-3-5-sonnet', help='Version of the model')
parser.add_argument('--mask_fix_position', type=int, default= 0, help='The total number of masked sentence is 2N')
parser.add_argument('--per_num_mask', type=int, default= 1, help='The per number of masked word')
parser.add_argument('--gpu', type=int, default= 0, help='Device to use (e.g., cuda:0, cuda:1, cpu)')
parser.add_argument('--max_new_tokens', type=int, default= 15, help='The number of max generated token')
parser.add_argument('--max_length', type=int, default= 0, help='The number of max generated token')
parser.add_argument('--n_split', type=int, default= 20, help='The number of max generated token')
parser.add_argument('--interv_type', type=str, default='mask', help='The type of intervention')
parser.add_argument('--log_file_path', default= 'acc_claude_claude-3-5-sonnet-20240620_0913_2115', help='The path of checking the correction')


parser.add_argument('--dataset_name', type=str, default='2_digit_multiplication', help='Name of the dataset to load')
parser.add_argument('--num_mask', type=int, default= 2, help='The total number of masked sentence is 2N')
parser.add_argument('--prompt', type=str, default= 'superhigh', help='The level of prompts')


args = parser.parse_args()
log_file_path = f'./results/{args.dataset_name}/results_{args.prompt[0]}/{args.log_file_path}/running.log'

print('mask_fix_position: ', args.mask_fix_position)
if args.dataset_name == 'word_unscrambling':
    args.mask_fix_position = 2

def parse_log_file(log_file_path):
    valid_indices = []
    line_index = 0
    all_count = 0
    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            if "prediction:" in line and "label:" in line:
                all_count +=1
                if args.dataset_name == 'word_unscrambling':
                    parts = line.strip().split('output_flags:')
                    main_parts = parts[0].strip().split(', label:')
                    prediction = main_parts[0].split(':')[1].strip()
                    label_str = main_parts[1].strip()
                    label = label_str.strip().rstrip(',')
                    if (isinstance(label, str) and label == prediction) or (prediction in label):
                        valid_indices.append(line_index)
                else:
                    parts = line.strip().split(',')
                    prediction = parts[0].split(':')[1].strip()
                    label = parts[1].split(':')[1].strip()
                    if float(prediction) == float(label):
                        valid_indices.append(line_index)
                line_index += 1
    return valid_indices


def extract_data_loader_by_indices(data_loader, valid_indices):
    dataset = data_loader.dataset
    new_dataset = torch.utils.data.Subset(dataset, valid_indices)
    new_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=data_loader.batch_size, shuffle=False)
    return new_data_loader

def create_new_dataloader(data_loader, num_to_mask, tokenizer, args):
    new_texts = []
    new_labels = []
    new_flags = []

    valid_indices = parse_log_file(log_file_path)
    data_loader_ = extract_data_loader_by_indices(data_loader, valid_indices)

    if args.interv_type in ['mask', 'replace', 'swapping']:
        for idx, (texts, labels) in enumerate(tqdm(data_loader_, desc="Processing DataLoader")):
            # print('texts, labels:', texts, labels)
            dataset = MaskedWordDataset(texts, labels, idx, num_to_mask, tokenizer, interv_type=args.interv_type, args=args)
            masked_texts, masked_labels, masked_flags = dataset.create_masked_dataset()
            # print(masked_texts, masked_labels, masked_flags)
            new_texts.extend(masked_texts)
            new_labels.extend(masked_labels)
            new_flags.extend(masked_flags)
        new_dataset = MaskedWordDataset(new_texts, new_labels, new_flags, num_to_mask, tokenizer, interv_type=args.interv_type, args=args)
    new_data_loader = DataLoader(new_dataset, batch_size=5, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer))
    return new_data_loader




"""
Load Model and Data and Intervention
"""
tokenizer = None
data_loader = load_dataloader(args)
print(Fore.RED + Style.BRIGHT + f"Original Data Size is {len(data_loader.dataset)}" + Style.RESET_ALL) 
data_loader = create_new_dataloader(data_loader, args.num_mask, tokenizer, args)
print(Fore.RED + Style.BRIGHT + f"Interventioned Data Size is {len(data_loader.dataset)}" + Style.RESET_ALL) 
if len(data_loader.dataset) % (args.num_mask*2 + 1) != 0:
    error_message = (
        f"Error: Dataset length (len(data_loader.dataset)) is not divisible by {args.num_mask*2 + 1}. "
        f"Please adjust the dataset size or num_mask value."
    )
    raise ValueError(error_message)
"""
Evaluation and Extract activations
"""
args.folder_path = f'./results/{args.dataset_name}/results_{args.prompt[0]}_interv/{args.log_file_path}_intev{args.interv_type}'
os.makedirs(args.folder_path, exist_ok=True)
logger = create_and_get_logger(args.folder_path)
llm = llm_agent(args.model_name, args.version, data_loader,args, logger, args.folder_path)
if args.max_length==0:
    print(Fore.GREEN + Style.BRIGHT + "Evaluation of LLM" + Style.RESET_ALL) 
    batch_id = llm.evaluate_accuracy()
    if batch_id == -1:
        print("all evaluated")
    else:
        print(f"Evaluation stopped at batch {batch_id}")
