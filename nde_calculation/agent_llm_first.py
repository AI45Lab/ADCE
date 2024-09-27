import datetime
import argparse
import os
import warnings
import sys
import os
sys.path.append('../')
sys.path.append('./')
from utils.load_data import *
from utils.load_model import *
from utils.lm_hf import *
from utils.util import *

from utils.agent_utils import llm_agent

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Load and process datasets.")
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for DataLoader')
parser.add_argument('--shuffle', type=bool, default=False, help='Whether to shuffle the dataset')
parser.add_argument('--num_workers', type=int, default=4, help='The number of loading workers')
parser.add_argument('--gpu', type=int, default= 0, help='Device to use (e.g., cuda:0, cuda:1, cpu)')
parser.add_argument('--max_new_tokens', type=int, default= 20,help='The number of max generated token')
parser.add_argument('--folder_path', type=str, help='The folder path',default=None)
parser.add_argument('--model_name', type=str, default='claude', help='Name of the model to load')
parser.add_argument('--version', type=str, default='claude-3-5-sonnet-20240620', help='Version of the model')

parser.add_argument('--dataset_name', type=str, default='analytic_entailment', help='Name of the dataset to load')
parser.add_argument('--prompt', type=str, default= 'superhigh', help='The level of prompts')


args = parser.parse_args()




"""
Load Model and Data
"""
data_loader = load_dataloader(args)

"""
Evaluation and Extract activations
"""
if args.folder_path is None:
    now = datetime.datetime.now()
    current_time = now.strftime("%m%d_%H%M")
    args.folder_path =  f"./results/{args.dataset_name}/results_{args.prompt[0]}/acc_{args.model_name}_{current_time}"
    os.makedirs(args.folder_path, exist_ok=True)
logger = create_and_get_logger(args.folder_path)
llm = llm_agent(args.model_name, args.version, data_loader,args, logger, args.folder_path)
batch_id = llm.evaluate_accuracy()
if batch_id == -1:
    print("all evaluated", flush = True)
else:
    print(f"Evaluation stopped at batch {batch_id}", flush = True)
