# add api key here
import os
os.environ['ANTHROPIC_API_KEY'] = "SET_YOUR_API_HERE"
os.environ['OPENAI_API_KEY'] = "SET_YOUR_API_HERE"


import warnings
warnings.filterwarnings('ignore')

import os
import random
import torch
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)


try:
    from utils.load_data import load_datasets_config
except:
    from load_data import load_datasets_config



from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.policies import apply_fsdp_checkpointing

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from transformers.tokenization_utils_base import BatchEncoding

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
)

from llama_recipes.utils.train_utils import (
    # train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from accelerate.utils import is_xpu_available

import json

from datasets import Dataset
import numpy as np


from transformers.data import DataCollatorWithPadding, DataCollatorForSeq2Seq

from tqdm import tqdm

import sys
# sys.path.append('./utils')
from utils.lm_hf import llm

from baukit import TraceDict


import re




from anthropic import Anthropic, RateLimitError
from openai import OpenAI


import builtins

def print(*args, **kwargs):
    if 'flush' not in kwargs:
        kwargs['flush'] = True
    builtins.print(*args, **kwargs) 


def update_config_inplace(train_config, fsdp_config, model_name,args):
    train_config.model_name = model_name
    train_config.use_peft = True
    train_config.quantization = True
    train_config.gradient_accumulation_steps = 1
    train_config.batch_size_training = 20
    train_config.lr = 1e-3
    train_config.context_length = 512
    train_config.seed = 42


    train_config.select_data = False
    train_config.select_num = None


    train_config.dataset_name = args.dataset_name

    train_config.batching_strategy = 'padding'
    train_config.num_labels = 2
    train_config.dataset_path = load_datasets_config().get(train_config.dataset_name, None)
    
    
    train_config.enable_fsdp = False
    fsdp_config.pure_bf16 = False    

    return train_config, fsdp_config




def set_config(**kwargs):
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    # we can update the config in the script
    train_config, fsdp_config = update_config_inplace(train_config, fsdp_config, kwargs['model_name'], args = kwargs['args'])
    # or we can update the config from CLI, it will overwrite the above settings
    update_config((train_config, fsdp_config), **kwargs)
    
    return train_config, fsdp_config


def load_model_classification(model_name,peft_model_id, args):
    train_config, fsdp_config = set_config(model_name = model_name, args = args)

    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.quantization:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(model_name if train_config.tokenizer_name is None else train_config.tokenizer_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if args.generate:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            use_cache=use_cache,
            pad_token_id = tokenizer.eos_token_id,
            output_attentions=True,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            use_cache=use_cache,
            num_labels = train_config.num_labels,
            pad_token_id = tokenizer.eos_token_id,
            output_attentions=True,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        )

    model.load_adapter(peft_model_id)

    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))


    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)


    hsdp_device_mesh = None
    if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
        hsdp_device_mesh = hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size, sharding_group_size=fsdp_config.sharding_group_size)
        print("HSDP device mesh is ready")

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(model, train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

        device_id = 0
        if is_xpu_available():
            device_id = torch.xpu.current_device()
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=(lambda module: module.to_empty(device=torch.device("cuda"), recurse=False))
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to("cuda")

    return model, tokenizer, train_config



def get_processed_dataset(dataset_path, tokenizer, group_size,train_split_ratio, add_prompt=False, dataset_name = None):
    def tokenize_add_label_train(sample):
        input = tokenizer.encode(tokenizer.bos_token + sample["text"], add_special_tokens=False)
        output = tokenizer.encode(sample["labels"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": input + output,
            "attention_mask" : [1] * (len(input) + len(output)),
            "labels": [-100] * len(input) + output,
            }

        return sample

    def tokenize_add_label_test(sample):
        input = tokenizer.encode(tokenizer.bos_token + sample["text"], add_special_tokens=False)
        output = tokenizer.encode(sample["labels"], add_special_tokens=False)
        sample = {
            "input_ids": input,
            "attention_mask" : [1] * len(input),
            "labels": output,
            }

        return sample
    
    def preprocess_function(examples):
        if add_prompt:
            input_ori = [(f"\nAs an expert in linguistic entailment, you will be provided with two sentences and determine if there is an entailment relationship between sentence 1 and sentence 2. An entailment relationship exists when the truth of sentence 1 guarantees the truth of sentence 2.\n  \n### Sentences:\n{{input}}\n \n### Relation (entailment or no-entailment):\n").format(input=examples["text"][i]) for i in range(len(examples["text"]))]
        else:
            input_ori = examples["text"]
        return tokenizer(input_ori, truncation=True,
            max_length= 98,
            padding='max_length',
            add_special_tokens=True)

    
    if 'civil' in dataset_name:
        data = Dataset.from_json(dataset_path, field='examples')#load_dataset('json', data_files=dataset_path, field='examples')['train']
    elif 'entail' in dataset_name:
        data = Dataset.from_json(dataset_path)['examples'][0]
    num_samples = len(data)
    # print('len(data)',len(data))
    num_groups = num_samples // group_size + (1 if num_samples % group_size != 0 else 0)
    # print('num_groups:',num_groups)
    groups = [data[i*group_size:(i+1)*group_size] for i in range(num_groups)]
    if train_split_ratio!=0:
        train_groups = groups[:int(num_groups * train_split_ratio)]
        if 'civil' in dataset_name:
            train_data = [{'input': input_text, 'toxicity': toxicity} for group in train_groups for input_text, toxicity in zip(group['input'], group['toxicity'])]
        elif 'entail' in dataset_name:
            train_data = [item for group in train_groups for item in group]
        
    test_groups = groups[int(num_groups * train_split_ratio):]
    # print('int(num_groups * train_split_ratio):',len(groups),int(num_groups * train_split_ratio))
    # print('train_groups :',len(train_groups))
    # print('test_groups :',len(test_groups))
    # train_data = [item for group in train_groups for item in group]
    if 'civil' in dataset_name:
        try:
            test_data = [{'input': input_text, 'toxicity': toxicity} for group in test_groups for input_text, toxicity in zip(group['input'], group['toxicity'])]
        except:
            test_data = [{'input': input_text, 'toxicity': toxicity} for group in test_groups for input_text, toxicity in zip(group['comment_text'], group['toxicity'])]
    elif 'entail' in dataset_name:
        test_data = [item for group in test_groups for item in group]
    # print('train_groups.shape:',len(train_data))
    print('test_data.shape:',test_data[0])
    dataset = {}
    prompt = (
        f"Input:\n{{input_text}}\nOutput:\n"
    )
    if train_split_ratio!=0:
        if 'civil' in dataset_name:
            dataset['train'] = Dataset.from_dict({
                "text": [item['input'] for item in train_data],
                "label": [0 if item['toxicity'] == 0 else 1 for item in train_data]
            }).map(preprocess_function, batched=True).remove_columns(['text'])
        elif 'entail' in dataset_name:
            dataset['train'] = Dataset.from_dict({
                "text": [item['input'] for item in train_data],
                "label": [0 if item['target_scores']['entailment'] == 1 else 1 for item in train_data]
            }).map(preprocess_function, batched=True).remove_columns(['text'])

    if 'civil' in dataset_name:
        dataset['test'] = Dataset.from_dict({
            "text": [item['input'] for item in test_data],
            "label":  [0 if item['toxicity'] == 0 else 1 for item in test_data]
        }).map(preprocess_function, batched=True).remove_columns(['text'])
    elif 'entail' in dataset_name:
        dataset['test'] = Dataset.from_dict({
            "text": [item['input'] for item in test_data],
            "label": [0 if item['target_scores']['entailment'] == 1 else 1 for item in test_data]
        }).map(preprocess_function, batched=True).remove_columns(['text'])

    # print('train_groups.shape:',dataset['train'].shape)
    # print('test_data.shape:',dataset['test'].shape)
    return dataset


class DatasetConfig:
    def __init__(self, dataset_path, num_workers_dataloader = 1, batch_size_training = 20):
        self.dataset_path = dataset_path
        self.num_workers_dataloader = num_workers_dataloader
        self.batch_size_training = batch_size_training

def load_dataloader(train_config, tokenizer,args):
    if args.interv:
        group_size = args.group_size
        # print('train_config.dataset_path:',train_config.dataset_path)
    else:
        group_size = 1
    dataset = get_processed_dataset(train_config.dataset_path, tokenizer,group_size,train_split_ratio=args.train_split_ratio, dataset_name = args.dataset_name)
    if args.train_split_ratio!=0:
        dataset_train = dataset['train']
    dataset_val = dataset['test']
    # return dataset_train, dataset_val

    # Create DataLoaders for the training and validation dataset

    if args.train_split_ratio!=0:
        train_dataloader = torch.utils.data.DataLoader(
            dataset_train,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            collate_fn = DataCollatorWithPadding(tokenizer=tokenizer),
            batch_size = train_config.batch_size_training
        )

    eval_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer),
        batch_size = 5
    )
    
    if args.train_split_ratio!=0:
        return train_dataloader, eval_dataloader
    else:
        return None,eval_dataloader

def client_generate_response(model_choice, version_choice, prompt_system, prompt_user, additional_content = []):
    
    # determining agent
    if model_choice == 'claude':
        client = Anthropic()
        messages=[
            {"role": "user", "content": [{"type": "text","text": prompt_user + ""}]}
        ]
        # adding chat information
        if len(additional_content)>0:
            for content in additional_content:
                messages.append(
                    {"role": content['role'], "content": [{"type": "text", "text": content['text']}]}
                )
    elif model_choice == 'gpt':
        client = OpenAI()
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user}
        ]
        # adding chat information
        if len(additional_content)>0:
            for content in additional_content:
                messages.append(
                    {"role": content['role'], "content": content['text']}
                )
    # request
    message_text = None
    for _ in range(3):
        try:
            if model_choice == 'claude':
                output = client.messages.create(
                    # model="claude-3-5-sonnet-20240620",
                    # model = "claude-3-opus-20240229",
                    model = version_choice,
                    max_tokens=1000,
                    temperature=0,
                    system = prompt_system,
                    messages = messages
                )
                message_text = output.content[-1].text

            elif model_choice == 'gpt':
                output = client.chat.completions.create(
                    # model="gpt-4o-mini",
                    model = version_choice,
                    messages = messages
                )
                message_text = output.choices[0].message.content
                
            break
        except Exception as e:
            print(f"An error occurred: {e}", flush=True)
    return message_text

def load_new_prompts():
    with open('../model_data_config_agent.json', 'r') as f:
        prompts = json.load(f)
    return prompts

class llm_agent(llm):
    def __init__(self, model, version, data_loader,args, logger,folder_path):
        self.data_loader = data_loader
        self.dataset_name = args.dataset_name
        self.model = model
        self.version = version
        self.model_name = args.model_name
        self.logger = logger
        self.folder_path = folder_path
        self.propmt = args.prompt
        self.args = args

        prompts = load_new_prompts()

        self.PROMPT_TEMPLATE = prompts['prompt'][self.dataset_name][self.propmt]
        
    def get_responses(self, statements,labels):
        all_generations = []
        outputs = []
        for statement,label in zip(statements,labels):
            statement = self.process_statements(statement,label)
            output = client_generate_response(self.model, self.version, "", statement, additional_content = [])
            all_generations.append(output)
            
        for seq in all_generations:
            res = self.format_response(seq)
            if (self.dataset_name =='2_digit_multiplication' or self.dataset_name =='GSM8k') and res is not None:
                res =  self.convert_to_number(res)
            outputs.append(res)
        del statements
        torch.cuda.empty_cache()
        return outputs
    
    def format_response(self, decoded_output):
        if self.dataset_name == 'empirical_judgments':
            match = re.search(r'### (Response|Relation|Answer causal or correlative or neutral|Answer 1 or 2 or 3):\s*(\w+)',decoded_output)
            return match.group(2).lower() if match else None
        
        elif self.dataset_name in ['2_digit_multiplication', 'GSM8k']:
            pattern = r'The answer is:?\s*\$?(-?[\d,.]+)(?:\s*[a-zA-Z]+\.?)?'
            matches = list(re.finditer(pattern, decoded_output, re.IGNORECASE | re.DOTALL))
            if matches:
                last_match = matches[-1]
                pre = last_match.group(1)
                print(f"pre: {pre}")
                return pre
            print('pre: -2 ')
            return -2
        elif self.dataset_name =='analytic_entailment':
            pattern = r"\b(entailment|no[-\s]?entailment|noentailment|entail|entails)\b"
            matches = list(re.finditer(pattern, decoded_output, re.IGNORECASE | re.DOTALL))#re.search(pattern, decoded_output, re.IGNORECASE)
            if matches:
                match = matches[-1]
                pre = match.group(1).lower() if match else None
                relation_index = decoded_output.find("### Relation:")
                if relation_index!= -1:
                    relation_part = decoded_output[relation_index:]
                    if 'entailment' in relation_part.lower() and 'no-entailment' in relation_part.lower():
                        return -2
                if pre.lower() == "entailment" or pre.lower() == "entail" or pre.lower() == "entails":
                    return 0
                elif pre.lower()  == "no-entailment" or pre.lower()  == "noentailment" or pre.lower()  == "no entailment":
                    return 1
            return -2

        elif self.dataset_name == 'word_unscrambling':
            pattern = r"The unscrambled word is\s*:?\s*([\w']+)"
            matches = list(re.finditer(pattern, decoded_output, re.DOTALL))
            
            if matches:
                last_match = matches[-1]
                pre = last_match.group(1)
            else:
                pre = None

            if pre is None:
                return -2
            else:
                pre = self.extract_words(pre)
                return pre.lower()

        elif self.dataset_name in ['general_knowledge', 'commonsenseqa', 'mmlu', 'tracking', 'mmlu-math']:

            list_pattern = r"The answer is:?\s*((\([a-zA-Z]\)\s*)+$|(\n\([a-zA-Z]\)[^\n]*){2,})"
            if re.search(list_pattern, decoded_output, re.DOTALL):
                print("pre: -2")
                return -2

            answer_pattern = r"answer is(?:\s*[:\s]*)\(([a-zA-Z])\)"
            matches = list(re.finditer(answer_pattern, decoded_output, re.DOTALL | re.MULTILINE))
            
            if matches:
                last_match = matches[-1]
                answer_option = last_match.groups()[0]
                pre = f"({answer_option.lower()})"
                print("pre:", pre)
                return pre

            print("pre: -2")
            return -2

        elif self.dataset_name =='civilcomments' or self.dataset_name =='civilcomments_min':
            pattern = r"(?i)### Response:\s*['\"]?(the text is (toxic|no-toxic|notoxic|no toxic))['\"]?"
            match = re.search(pattern, decoded_output, re.IGNORECASE)
            pre = match.group(2).lower() if match else None
            relation_index = decoded_output.find("### Response:")
            print('pre:',pre)
            if relation_index!= -1:
                relation_part = decoded_output[relation_index:]
                if 'toxic' in relation_part.lower() and 'no-toxic' in relation_part.lower() and 'no toxic' in relation_part.lower():
                    return -2 #None
            if pre is None:
                return -2 #None
            if pre.lower() == "toxic": #or pre.lower() == "entail" or pre.lower() == "entails":
                return 1
            elif pre.lower()  == "no-toxic" or pre.lower()  == "no toxic" or pre.lower()  == "notoxic":
                return 0
    
    def evaluate_accuracy(self):
        all_outputs, all_labels = [], []
        for batch_idx, batch in enumerate(tqdm(self.data_loader)):
            if len(batch) == 2 and (self.propmt == 'superhigh' or self.propmt == 'fewshot_superhigh' ):
                input_texts, labels = batch
                output_flags=input_texts
                options = labels
            elif len(batch) == 3 and (self.propmt == 'superhigh' or self.propmt == 'fewshot_superhigh' ):
                input_texts, labels,output_flags = batch
                options = labels
                # all_output_flags.extend(output_flags)
            elif len(batch) == 3 and (self.propmt == 'csuperhigh' or self.propmt == 'fewshot_csuperhigh' ):
                input_texts, options,labels = batch
                output_flags=input_texts
            elif len(batch) == 4 and (self.propmt == 'csuperhigh' or self.propmt == 'fewshot_csuperhigh' ):
                input_texts, options,labels,output_flags = batch
                # all_output_flags.extend(output_flags)
            try:
                outputs = self.get_responses(input_texts, options)
            except:
                return batch_idx
            for i in range(len(outputs)):
                log_message = f"prediction: {outputs[i]}, label: {list(labels)[i]}, output_flags: {output_flags[i]}"
                self.logger.info(log_message)
            all_labels.extend(list(labels))
            all_outputs.extend(outputs)
        self.logger.info(f"PROMPT_TEMPLATE is \n  {self.PROMPT_TEMPLATE}\n")
        self.logger.info(f"All Outputs for Model {self.model_name} on Dataset {self.dataset_name}:\n")
        correct_predictions = 0
        for output, label in zip(all_outputs, all_labels):
            if self.compare_values(label, output):
                correct_predictions += 1
        print(f"All Outputs for Model {self.model_name} on Dataset {self.dataset_name} has been stored.\n")
        print('correct_predictions:',correct_predictions,len(all_labels))
        accuracy = 100*correct_predictions / len(all_labels)
        self.logger.info(f"Prediction accuracy for Model {self.model_name} on Dataset {self.dataset_name}: {accuracy:.6f}%")
        print(f"Prediction accuracy for Model {self.model_name} on Dataset {self.dataset_name}: {accuracy:.6f}%")
        return -1
