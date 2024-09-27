import re
import torch
from tqdm import tqdm
import logging
import datetime
import argparse
import numpy as np
import random
import sys
import os
from functools import partial
# sys.path.append('./utils')
from utils.load_data import *
from utils.load_model import *
import copy
import gc
from baukit import TraceDict

with open('../model_data_config.json', 'r') as f:
  prompts = json.load(f)




def create_prompt(PROMPT_TEMPLATE,instruction: str,args) -> str:
    if args.dataset_name == 'analytic_entailment':
        sentences = instruction.split('.',1)
        formatted_sentences = []
        for i, s in enumerate(sentences, 1):
            s = s.strip()  
            if s:
                formatted_sentences.append(f"sentence {i}: {s}")
            else:
                formatted_sentences.append(f"sentence {i}: ")
        formatted_output = '\n'.join(formatted_sentences)
        prompt = PROMPT_TEMPLATE.replace("[INSTRUCTION]", formatted_output)
    else:
        prompt = PROMPT_TEMPLATE.replace("[INSTRUCTION]", instruction)
    return prompt
    

def create_prompt_mc(PROMPT_TEMPLATE, instruction: str, options: dict,args) -> str:
    if args.dataset_name == 'analytic_entailment':
        sentences = instruction.split('.',1)
        formatted_sentences = []
        for i, s in enumerate(sentences, 1):
            s = s.strip()  
            if s:
                formatted_sentences.append(f"sentence {i}: {s}")
            else:
                formatted_sentences.append(f"sentence {i}: ")
        formatted_output = '\n'.join(formatted_sentences)
        prompt = PROMPT_TEMPLATE.replace("[INSTRUCTION]", formatted_output)
    else:
        prompt = PROMPT_TEMPLATE.replace("[INSTRUCTION]", instruction)
    options_text = "\n".join([f"{key} {value}" for i, (key,value) in enumerate(options.items())])
    last_key = list(options.keys())[-1]
    prompt = prompt.replace("[OPTIONS]",options_text)
    prompt = prompt.replace("[LASTOPTIONS]",last_key)
    return prompt


class llm():
    def __init__(self,model, tokenizer,data_loader,args, logger,folder_path):
        self.data_loader = data_loader
        self.dataset_name = args.dataset_name
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = args.model_name
        self.device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
        self.max_new_tokens = args.max_new_tokens
        self.logger = logger
        self.folder_path = folder_path
        self.propmt = args.prompt
        self.args = args
        if "commonsense" in self.dataset_name:
            with open('../model_data_config_agent.json', 'r') as f:
                prompts = json.load(f)
        else:
            with open('../model_data_config.json', 'r') as f:
                prompts = json.load(f)
        PROMPT_TEMPLATE = prompts['prompt'][self.dataset_name][self.propmt]
        self.PROMPT_TEMPLATE = PROMPT_TEMPLATE


    def extract_words(self,texts):
        words = []
        for text in texts:
            match = re.findall(r'\b\w+\b', text)
            words.extend(match)
        return "".join(words)

    def format_response(self, sequence, statement = None):
        decoded_output = self.tokenizer.decode(sequence, skip_special_tokens=True)
        print('decoded_output:',decoded_output)
        if self.dataset_name == 'empirical_judgments':
            match = re.search(r'### (Response|Relation|Answer causal or correlative or neutral|Answer 1 or 2 or 3):\s*(\w+)',decoded_output)
            return match.group(2).lower() if match else None
        elif self.dataset_name in ['2_digit_multiplication', 'GSM8k']:
            pattern = r'### Response:\s*The answer is:?\s*\$?(-?[\d,.]+)(?:\s*[a-zA-Z]+\.?)?'
            matches = list(re.finditer(pattern, decoded_output, re.IGNORECASE | re.DOTALL))
            if matches:
                last_match = matches[-1]
                pre = last_match.group(1)
                print(f"pre: {pre}")
                return pre
            print('pre: -2 ')
            return -2
        elif self.dataset_name =='analytic_entailment':
            pattern = r"(?i)### Relation \(entailment or no-entailment\):\n.*?(entailment|no-entailment|no entailment|noentailment|entail|entails)" 
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
            pattern = r"### Output:\s*The unscrambled word is\s*:?\s*([\w']+)"
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
            list_pattern = r"Among \(a\) through \(([a-zA-Z])\), the answer is:?\s*((\([a-zA-Z]\)\s*)+$|(\n\([a-zA-Z]\)[^\n]*){2,})"
            if re.search(list_pattern, decoded_output, re.DOTALL):
                print("pre: -2")
                return -2
            if 'mistral' in self.model_name or 'mixtral' in self.model_name:
                answer_pattern = r"answer is(?:\s*[:\s]*)\(([a-zA-Z])\)"
            else:
                answer_pattern = r"Among \(a\) through \(([a-zA-Z])\), the answer is:?\s*\(([a-zA-Z])\)(?:\s*:?\s*(?![\(a-zA-Z\)\n]+$).*?)"
            matches = list(re.finditer(answer_pattern, decoded_output, re.DOTALL | re.MULTILINE))
                
            if matches:
                last_match = matches[-1]
                try:
                    _, answer_option = last_match.groups()
                except:
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
            
    def process_statements(self,instruction,labels):
        if self.propmt == 'csuperhigh' or self.propmt == 'fewshot_csuperhigh':
            return create_prompt_mc(self.PROMPT_TEMPLATE,instruction, labels,self.args)
        elif self.propmt == 'superhigh' or self.propmt == 'fewshot_superhigh':
            return create_prompt(self.PROMPT_TEMPLATE,instruction,self.args)
    
    def convert_to_number(self, s):
        if isinstance(s, (int, float)):
            return s
        if isinstance(s, str):
            s = s.strip()  
            if not s:  
                return -2  

            s = s.replace(',', '') 

            try:
                return int(s)
            except ValueError:
                if '.' in s:
                    try:
                        return float(s)  
                    except ValueError:
                        pass 
                cleaned = ''.join(c for c in s if c.isdigit() or c == '-')
                if cleaned:
                    return int(cleaned)  
                else:
                    return s 
        return s 
  
    
    def get_responses(self, statements,labels,max_new_tokens):
        all_generate_ids = []
        max_input_ids = 0
        outputs = []
        all_statements = []
        for statement,label in zip(statements,labels):
            statement = self.process_statements(statement,label)
            all_statements.append(statement)
            encoded_input = self.tokenizer(statement, return_tensors="pt")
            input_ids = encoded_input['input_ids'].to(self.device)
            if input_ids.shape[1] > max_input_ids:
                max_input_ids = input_ids.shape[1]
            attention_mask = encoded_input['attention_mask'].to(self.device)
            with torch.no_grad():
                generate_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    attention_mask=attention_mask,
                    return_dict_in_generate=True,
                    output_scores=True)
            all_generate_ids.append(generate_ids.sequences[0])
            del statement,label,encoded_input,generate_ids,input_ids,attention_mask
            torch.cuda.empty_cache()
            gc.collect()
            
        for seq,statement in zip(all_generate_ids, all_statements):
            res = self.format_response(seq,statement)
            if (self.dataset_name =='2_digit_multiplication' or self.dataset_name =='GSM8k') and res is not None:
                res =  self.convert_to_number(res)
            outputs.append(res)
        del statements,all_generate_ids
        torch.cuda.empty_cache()
        return outputs, max_input_ids

    def compare_values(self,label, output):
        if output is None:
            return False
        if hasattr(label, 'item'):
            label = label.item()
        if isinstance(label, list):
            return output in label
        if isinstance(label, tuple):
            label = label[0] if output else ''
        try:
            return float(label) == float(output)
        except ValueError:
            return str(label).lower().strip() == str(output).lower().strip()

    def evaluate_accuracy(self):
        all_outputs, all_labels, all_output_flags = [], [], []
        for batch in tqdm(self.data_loader):
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
            
            outputs, max_input_ids = self.get_responses(input_texts, options,self.max_new_tokens)
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
        return max_input_ids
