from torch.utils.data import Dataset, DataLoader

import re
import json
import os

# %%
import os
# 
os.environ['ANTHROPIC_API_KEY'] = "SET_YOUR_API_HERE"

from anthropic import Anthropic, RateLimitError
import time


def claude_client(prompt_system, prompt_user, additional_content = []):
    client = Anthropic()
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text","text": prompt_user}
            ]
        }
    ]
    if len(additional_content)>0:
        for content in additional_content:
            messages.append(
                {
                "role": content['role'],
                "content": [
                    {"type": "text", "text": content['text']}
                    ]
                }
            )
    for _ in range(3):
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                # model = "claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                system = prompt_system,
                messages = messages
            )
            break
        except RateLimitError as e:
            print(f"Rate limit exceeded: {e}", flush=True)
            print('Usage: ',message.usage, flush=True)
            time.sleep(60)
            
        except Exception as e:
            print(f"An error occurred: {e}", flush=True)
            break
    return message

# %%
class CommonsenseQADataset(Dataset):
    def __init__(self, file_path):
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                processed_item = self.process_item(item)
                data.append(processed_item)
        return data

    def process_item(self, item):
        question = item['question']
        input_text = question['stem']
        choices = question['choices']
        
        target_scores = {}
        for choice in choices:
            label = f"({choice['label'].lower()})"  
            text = choice['text']
            target_scores[label] = text

        return {
            'input': input_text,
            'target_scores': target_scores,
            'answerKey': f"({item['answerKey'].lower()})"  
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        target_scores = item['target_scores']
        targets = item['answerKey']
        return input_text, target_scores, targets 

# %%
def save_intervention(dataset_name, data_ori, data_flip, data_unflip):
    def save_json(fname, data):
        if not os.path.exists(fname+'.json'):
            # write intervention data
            with open(fname+'.json', 'w') as json_file:
                json.dump(data, json_file, indent=4)
        else:
            try:
                # If not empty, update data
                with open(fname+'.json', "r") as json_file:
                    old_data = json.load(json_file)
                    old_data += data
            except:
                # If empty
                old_data = data
            with open(fname+'.json', 'w') as json_file:
                json.dump(old_data, json_file, indent=4)

    f_ori = dataset_name+'_ori'
    if not isinstance(data_ori,list):
        data_ori = [data_ori,]
    save_json(f_ori, data_ori)

    f_flip = dataset_name+'_flip'
    if not isinstance(data_flip,list):
        data_flip = [data_flip,]
    save_json(f_flip, data_flip)

    f_unflip = dataset_name+'_unflip'
    if not isinstance(data_unflip,list):
        data_unflip = [data_unflip,]
    save_json(f_unflip, data_unflip)

# %%
def prompt_construction(task_config, input_text):
    if task_config.data == 'commonsenseqa':
        if task_config.task == 'rephrase_flip':
            prompt_system = (f"You are an expert in natural language processing and commonsense reasoning. " 
                            f"Your task is to rephrase the given commonsense question, and then modify the paraphrase " 
                            f"so that the modified question results in a different answer based on the provided options. " 
                            f"The input will be in the form of a dictionary: "
                            f"{{'Question':'question', 'Options':['option1', 'option2',...], 'Answer':'ans'}}, "
                            f"where 'Question' is the original commonsense question, 'Options' are the candidate answers, and 'Answer' is the original correct answer. "
                            f"Output only the modified Question without any introductory phrases."
                            )
            prompt_user = f"Here is the input: {{input}}. The modified question is:".format(input = input_text)
        elif task_config.task == 'rephrase_unflip':
            prompt_system = (f"You are an NLP and commonsense reasoning expert. "
                             f"Modify the keywords in the 'Question' to ensure the given 'Answer' is the most fitting answer to the modified result among the 'Options'. "
                             f"The input is in the form of a dictionary: {{'Question':'question', 'Options':['option1', 'option2', ...], 'Answer':'ans'}}. "
                             f"Output only the modified Question without any introductory phrases."
                             )
            prompt_user = f"Here is the input: {{input}}. The modified question is:".format(input = input_text)
        elif task_config.task == 'check':
            prompt_system = (
                            f"Below is a commonsense question along with some answer options. "
                             f"Choose the correct answer from these options. "
                             f"Your output should only be the answer enclosed in parenthesis, without any introductory phrases."
                            )
            prompt_user = f"{{input}}".format(input = input_text)
    return prompt_system, prompt_user

def format_input_intervention(task_config, input_corpus):
    input_str = ""
    if task_config.data == 'commonsenseqa':
        if task_config.task == 'rephrase_flip' or task_config.task == 'rephrase_unflip':
            for input_text in input_corpus:
                options = [key+' '+input_text['target_scores'][key] for key in input_text['target_scores'].keys()]
                input_text = f"{{'Question':'"+input_text['input']+"','Options':'"+str(options)+"','Answer':'"+str(input_text['answerKey'])+" "+input_text['target_scores'][input_text['answerKey']]+f"'}}"
                input_str += f"{{input}}".format(input = input_text)
        elif task_config.task == 'check':
            for input_text in input_corpus:
                prompt_template = f"### Question: [INSTRUCTION]\n[OPTIONS]\n \n### Among (a) through [LASTOPTION], the answer is"
                prompt = prompt_template.replace("[INSTRUCTION]", input_text['input'])
                options_text = "\n".join([f"{key} {value}" for i, (key,value) in enumerate(input_text['target_scores'].items())])
                input_str += prompt.replace("[OPTIONS]",options_text).replace("[LASTOPTION]", input_text['last_option'])
    return input_str

# %%
class PromptConfig:
    def __init__(self, data_description, multiple_input, use_fewshot = False):
        self.multiple_input = multiple_input
        self.use_fewshot = use_fewshot
        self.data = data_description

# %%
def get_data(file_path):
    def process_item(item):
        question = item['question']
        input_text = question['stem']
        choices = question['choices']
        
        target_scores = {}
        for choice in choices:
            label = f"({choice['label'].lower()})"  
            text = choice['text']
            target_scores[label] = text
            last_option = label

        return {
            'input': input_text,
            'target_scores': target_scores,
            'answerKey': f"({item['answerKey'].lower()})",
            'last_option':last_option
        }
    with open(file_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            item = json.loads(line.strip())
            processed_item = process_item(item)
            data.append(processed_item)
            # data.append(item)
    return data

# %%
def gen_rephrase(dataset, dataset_name, repeat_times, start_ind = 0, debug = False):
    task_config = PromptConfig(dataset_name, multiple_input = False, use_fewshot = False)
    for sample_id, data in enumerate(dataset):
        start_time = time.time()
        # flip
        new_data_flips = []
        additional_text = []
        flag_global = True
        for i in range(repeat_times):
            flag = False
            iter_time = 0
            while flag is not True:
                task_config.task = 'rephrase_flip'
                input_text = format_input_intervention(task_config, [data,])
                prompt_system, prompt_user = prompt_construction(task_config, input_text)
                message = claude_client(prompt_system, prompt_user, additional_text)
                if debug:
                    print('Usage: ',message.usage, flush=True)
                    print('Text: ',message.content[-1].text, flush=True)

                new_data_flip = data.copy()
                new_data_flip['input'] = message.content[-1].text
                new_data_flip['input_ori'] = data['input']
                
                additional_text.append({
                    'role': 'assistant',
                    'text': new_data_flip['input']
                })

                if i > 0 and new_data_flips[-1]['input'] == new_data_flip['input']:
                    additional_text.append({
                        'role': 'user',
                        'text': "The new modification is the same as the last one. Do not make the same modification. Please modify the question again. If it is hard for you, try to rephrase the last generation according to the requirement. Output only the modified Question."
                    })
                    iter_time += 1
                else:
                    # check
                    task_config.task = 'check'
                    input_text = format_input_intervention(task_config, [new_data_flip,])
                    prompt_system, prompt_user = prompt_construction(task_config, input_text)
                    message = claude_client(prompt_system, prompt_user)
                    if debug:
                        print('Usage: ',message.usage, flush=True)
                        print('Text: ',message.content[-1].text, flush=True)
                        
                    label = re.search(r'\((.*?)\)', data['answerKey']).group(1)
                    if len(message.content[-1].text) == 1:
                        pred = message.content[-1].text
                    else:
                        try:
                            pred = re.search(r'\((.*?)\)', message.content[-1].text).group(1)
                        except:
                            break
                    iter_time += 1

                    if pred != label:
                        flag = True
                        new_data_flip['answerKey_ori'] = data['answerKey']
                        new_data_flip['answerKey'] = '('+pred+')'
                    else:
                        additional_text.append({
                            'role': 'user',
                            'text': "The answer to the modified question is still the same as the original question. Please modify the question again. Output only the modified Question."
                        })

                if iter_time >= 10:
                    break
            if flag is not True:
                flag_global = False
                break
            new_data_flips.append(new_data_flip)
            additional_text.append({
                        'role': 'user',
                        'text': "Now, following the same requirements, modify the question again. The new modification should be different from the previous generation. Output only the modified Question."
                    })
        if flag_global is not True:
            continue
        # unflip
        new_data_unflips = []    
        additional_text = []
        for i in range(repeat_times):
            flag = False
            iter_time = 0
            additional_text = []
            while flag is not True:
                task_config.task = 'rephrase_unflip'
                new_data_unflip = new_data_flips[i].copy()
                new_data_unflip['answerKey'] = data['answerKey']

                input_text = format_input_intervention(task_config, [new_data_unflip,])
                prompt_system, prompt_user = prompt_construction(task_config, input_text)
                message = claude_client(prompt_system, prompt_user, additional_text)
                if debug:
                    print('Usage: ',message.usage, flush=True)
                    print('Text: ',message.content[-1].text, flush=True)

                new_data_unflip['input'] = message.content[-1].text
                additional_text.append({
                    'role': 'assistant',
                    'text': new_data_unflip['input']
                })

                
                if i > 0 and new_data_unflips[-1]['input'] == new_data_unflip['input']:
                    additional_text.append({
                        'role': 'user',
                        'text': "The new modification is the same as the last one. Do not make the same modification. Please modify the question again. If it is hard for you, try to rephrase the last generation according to the requirement. Output only the modified Question."
                    })
                    iter_time += 1
                else:
                    # check
                    task_config.task = 'check'
                    input_text = format_input_intervention(task_config, [new_data_unflip,])
                    prompt_system, prompt_user = prompt_construction(task_config, input_text)
                    message = claude_client(prompt_system, prompt_user)
                    if debug:
                        print('Usage: ',message.usage, flush=True)
                        print('Text: ',message.content[-1].text, flush=True)

                    
                    label = re.search(r'\((.*?)\)', data['answerKey']).group(1)
                    if len(message.content[-1].text) == 1:
                        pred = message.content[-1].text
                    else:
                        try:
                            pred = re.search(r'\((.*?)\)', message.content[-1].text).group(1)
                        except:
                            break
                    iter_time += 1

                    if pred == label:
                        flag = True
                        new_data_unflip['answerKey_ori'] = data['answerKey']
                    else:
                        additional_text.append({
                            'role': 'user',
                            'text': "The answer to the modified question is different from the original question. Please modify the question again. Output only the modified Question."
                        })
                if iter_time >= 10:
                    break
            if flag is not True:
                flag_global = False
                break
            new_data_unflips.append(new_data_unflip)
            additional_text.append({
                        'role': 'user',
                        'text': "Now, following the same requirements, modify the question again. The new modification should be different from the previous generation. Output only the modified Question."
                    })
        if flag_global is not True:
            continue
        save_intervention(dataset_name, data, new_data_flips, new_data_unflips)
        end_time = time.time() 
        print(f"Sample {sample_id + start_ind} intervened using {round((end_time - start_time)/60,4)} minutes.", flush=True)
            


if __name__ == "__main__":
    dataset = get_data(file_path = '../data/commonsenseqa/task.jsonl')
    dataset_name = 'commonsenseqa'  
    repeat_times = 2
    start_ind = 0
    gen_rephrase(dataset[start_ind:], dataset_name, repeat_times, start_ind = start_ind)
