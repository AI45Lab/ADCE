# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings('ignore')

import random
import torch
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)
import sys

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

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



sys.path.append("../")
sys.path.append("./")
from utils.load_data import load_datasets_config
from utils.load_model import load_models_config
from datasets import Dataset
import numpy as np
from transformers.data import DataCollatorWithPadding
from utils.newtrainer import train


# customize the config here 
def update_config_inplace(train_config, fsdp_config):
    train_config.model_name = "llama-3-8b"
    train_config.model_path = load_models_config().get(train_config.model_name, None)
    train_config.use_peft = True
    train_config.quantization = True
    train_config.use_fast_kernels = True
    train_config.use_fp16 = False

    
    train_config.from_peft_checkpoint = False
    train_config.run_validation = True
    train_config.save_model = True

    train_config.num_epochs = 100
    
    train_config.gradient_accumulation_steps = 1
    train_config.batch_size_training = 20
    train_config.lr = 3e-4
    train_config.context_length = 512
    train_config.seed = 42

    train_config.select_data = False
    train_config.select_num = None

    train_config.batching_strategy = 'padding'
    train_config.dataset = 'analytic_entailment_interv'
    train_config.dataset_path = load_datasets_config().get(train_config.dataset, None)
    train_config.num_labels = 2
    train_config.add_prompt = False

    train_config.continue_training = False
    train_config.continue_name = ''
    train_config.continue_epoch = 1
    
    train_config.output_dir = (\
        "ckpt/ckpt_"+train_config.model_name+\
        "_"+train_config.dataset+\
        "_batch-size-"+str(train_config.batch_size_training)+\
        "_accum-"+str(train_config.gradient_accumulation_steps)+\
        "_ctx-"+str(train_config.context_length)+\
        "_lr-"+str(train_config.lr)+\
        "_seed-"+str(train_config.seed)
        )


    if not train_config.add_prompt:
        train_config.output_dir += '_no-prompting'

    if train_config.continue_training:
        train_config.output_dir += '_continue'


    train_config.enable_fsdp = False
    fsdp_config.pure_bf16 = False    

    return train_config, fsdp_config


def get_processed_dataset(dataset_path, tokenizer, train_split_ratio = 0.6, add_prompt = True):
    def preprocess_function(examples):
        if add_prompt:
            input_ori = [(f"\nAs an expert in linguistic entailment, you will be provided with two sentences and determine if there is an entailment relationship between sentence 1 and sentence 2. An entailment relationship exists when the truth of sentence 1 guarantees the truth of sentence 2.\n  \n### Sentences:\n{{input}}\n \n### Relation (entailment or no-entailment):\n").format(input = examples["text"][i]) for i in range(len(examples["text"]))]
        else:
            input_ori = examples["text"]
        return tokenizer(input_ori, truncation=True)
    data = Dataset.from_json(dataset_path)['examples'][0]
    train_idx = np.arange(round(len(data)*train_split_ratio))
    test_idx = [i for i in range(len(data)) if i not in train_idx]

    dataset = {}
    dataset['test'] = Dataset.from_dict({"text":[data[i]['input'] for i in test_idx],"label":[0 if data[i]['target_scores']['entailment'] == 1 else 1 for i in test_idx]}).map(preprocess_function, batched = True).remove_columns(['text'])
    dataset['train'] = Dataset.from_dict({"text":[data[i]['input'] for i in train_idx],"label":[0 if data[i]['target_scores']['entailment'] == 1 else 1 for i in train_idx]}).map(preprocess_function, batched = True).remove_columns(['text'])

    return dataset

def evaluate_acc(model, eval_dataloader):
    import evaluate
    evaL_acc = evaluate.load("accuracy")

    for step, batch in enumerate(eval_dataloader):
        # stop when the maximum number of eval steps is reached
        for key in batch.keys():
            batch[key] = batch[key].to('cuda:0')
        # Ensure no gradients are computed for this scope to save memory
        with torch.no_grad():
            # Forward pass and compute loss
            outputs = model(**batch).logits
            predictions = torch.argmax(outputs, axis=1)
        return evaL_acc.compute(predictions = predictions, references=batch['labels'])

def main(**kwargs):
    # Get the default configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    # we can update the config in the script
    train_config, fsdp_config = update_config_inplace(train_config, fsdp_config)
    # or we can update the config from CLI, it will overwrite the above settings
    update_config((train_config, fsdp_config), **kwargs)
    # Set the seeds for reproducibility
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

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)


    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.quantization:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_path if train_config.tokenizer_name is None else train_config.tokenizer_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForSequenceClassification.from_pretrained(
        train_config.model_path,
        quantization_config=quant_config,
        device_map="auto",
        use_cache=use_cache,
        num_labels = train_config.num_labels,
        pad_token_id = tokenizer.eos_token_id
    )

    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    if train_config.continue_training:
        model.load_adapter(train_config.continue_name)
        print('continue training from epoch '+str(train_config.continue_epoch))
    else:
        # Prepare the model for int8 training if quantization is enabled
        if train_config.quantization:
            model = prepare_model_for_kbit_training(model)

        # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
        if train_config.enable_fsdp and fsdp_config.pure_bf16:
            model.to(torch.bfloat16)

        if train_config.use_peft:
            # Load the pre-trained peft model checkpoint and setup its configuration
            if train_config.from_peft_checkpoint:
                model = PeftModel.from_pretrained(model, train_config.from_peft_checkpoint, is_trainable=True)
                peft_config = model.peft_config()
            # Generate the peft config and start fine-tuning from original model
            else:
                peft_config = generate_peft_config(train_config, kwargs)
                peft_config.lora_dropout = 0.01
                model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            


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

     # Load and preprocess the dataset for training and validation
    dataset = get_processed_dataset(train_config.dataset_path, tokenizer, add_prompt=train_config.add_prompt)
    dataset_train = dataset['train']
    dataset_val = dataset['test']
    
    if train_config.select_data:
        dataset_train = dataset_train.select(range(train_config.select_num))
        dataset_val = dataset_val.select(range(train_config.select_num))

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Validation Set Length = {len(dataset_val)}")


    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer),
        batch_size = train_config.batch_size_training,
        shuffle=True
    )

    eval_dataloader = None
    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            collate_fn = DataCollatorWithPadding(tokenizer=tokenizer),
            batch_size = len(dataset_val),
            shuffle=True
        )
        if len(eval_dataloader) == 0:
            raise ValueError("The eval set size is too small for dataloader to load even one batch. Please increase the size of eval set.")
        else:
            print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")


    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
        train_config.continue_epoch
    )

    
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    main()
