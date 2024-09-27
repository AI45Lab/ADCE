
import argparse
from colorama import Fore, Style
import os
import warnings
from torch.utils.data import Dataset, DataLoader
import sys
import os
sys.path.append('../')
sys.path.append('./')
from utils.load_data import custom_collate_fn, load_dataloader
from utils.util import create_and_get_logger
from utils.agent_utils import llm_agent

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Load and process datasets.")
parser.add_argument('--batch_size', type=int, default=5, help='Batch size for DataLoader')
parser.add_argument('--shuffle', type=bool, default=False, help='Whether to shuffle the dataset')
parser.add_argument('--num_workers', type=int, default=2, help='The number of loading workers')
parser.add_argument('--model_name', type=str, default='claude', help='Name of the model to load')
parser.add_argument('--version', type=str, default='claude-3-5-sonnet-20240620', help='Version of the model')
parser.add_argument('--per_num_mask', type=int, default= 1, help='The per number of masked word')
parser.add_argument('--gpu', type=int, default= 0, help='Device to use (e.g., cuda:0, cuda:1, cpu)')
parser.add_argument('--max_new_tokens', type=int, default= 20, help='The number of max generated token')
parser.add_argument('--n_split', type=int, default= 10, help='The number of max generated token')
parser.add_argument('--log_file_path', default= 'acc_claude_claude-3-5-sonnet-20240620_0914_0018', help='The path of checking the correction')



# # analytic_entailment
parser.add_argument('--dataset_name', type=str, default='analytic_entailment', help='Name of the dataset to load')
parser.add_argument('--interv_type', type=str, default='rephrase', help='The type of intervention')
parser.add_argument('--num_mask', type=int, default= 2, help='The total number of masked sentence is 2N')
parser.add_argument('--prompt', type=str, default= 'superhigh', help='The level of prompts')

args = parser.parse_args()


log_file_path = f'./results/{args.dataset_name}/results_{args.prompt[0]}/{args.log_file_path}/running.log'


def parse_log_file(log_file_path):
    valid_indices = []
    line_index = 0
    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            if "prediction:" in line and "label:" in line:
                parts = line.strip().split(',')
                prediction = parts[0].split(':')[1].strip()
                label = parts[1].split(':')[1].strip()
                if prediction == label:
                    # print(prediction,label)
                    valid_indices.append(line_index)
            line_index += 1
    return valid_indices


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def extract_data_loader_by_indices(data_loader, indices):
    dataset = data_loader.dataset
    new_data = [dataset[i] for i in indices]
    new_dataset = CustomDataset(new_data)
    new_data_loader = DataLoader(new_dataset, batch_size=data_loader.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    return new_data_loader



def create_new_dataloader(data_loader, group_size=5):
    valid_indices = parse_log_file(log_file_path)
    # valid_indices = [i for i in range(10)]
    print('valid_indices:',valid_indices, flush = True)
    expanded_indices = []
    for group in valid_indices:
        start_index = group * group_size
        expanded_indices.extend(range(start_index, start_index + group_size))
    # print('expanded_indices:',expanded_indices)
    new_data_loader = extract_data_loader_by_indices(data_loader, expanded_indices)
    return new_data_loader


"""
Load Model and Data and Intervention
"""
data_loader_original = load_dataloader(args,dataset_name=args.dataset_name +'_interv')
data_loader = create_new_dataloader(data_loader_original)
print(Fore.RED + Style.BRIGHT + f"Interventioned Data Size is {len(data_loader.dataset)}" + Style.RESET_ALL, flush = True) 

if len(data_loader.dataset) % (args.num_mask*2 + 1) != 0:
    error_message = (
        f"Error: Dataset length (len(data_loader.dataset)) is not divisible by {args.num_mask*2 + 1}. "
        f"Please adjust the dataset size or num_mask value."
    )
    raise ValueError(error_message)

"""
Evaluation and Extract activations
"""
folder_path = f'./results/{args.dataset_name}/results_{args.prompt[0]}_interv/{args.log_file_path}_intev{args.interv_type}'
os.makedirs(folder_path, exist_ok=True)
logger = create_and_get_logger(folder_path)
llm = llm_agent(args.model_name, args.version, data_loader,args, logger, folder_path)
print(Fore.GREEN + Style.BRIGHT + "Evaluation of LLM" + Style.RESET_ALL, flush = True) 
batch_id = llm.evaluate_accuracy()
if batch_id == -1:
    print("all evaluated")
else:
    print(f"Evaluation stopped at batch {batch_id}")
