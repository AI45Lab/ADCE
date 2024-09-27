from torch.utils.data import Dataset, DataLoader
import json
import numpy as np



class EmpiricalJudgmentsDataset(Dataset):
    def __init__(self, file_path):
        data = self.load_data(file_path)['examples']
        self.data = data['examples']
    
    def load_data(self,file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = example['input']
        target_scores = example['target_scores']
        
        if target_scores["causal"] == 1:
            label = "causal"
        elif target_scores["correlative"] == 1:
            label = "correlative"
        elif target_scores["neutral"] == 1:
            label = "neutral"
        else:
            label = "NA"
        
        return input_text, label


class TwodigitMultiplicationDataset(Dataset):
    def __init__(self, file_path):
        data = self.load_data(file_path)
        self.data = data['examples']
    
    def load_data(self,file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input = item['input']
        label = item['target']
        target_scores = item['target_scores']
        return input,label

class NewRuleDataset(Dataset):
    def __init__(self, file_path, train_split_ratio = 0.6):
        data = self.load_data(file_path)
        train_idx = np.arange(round(len(data)*train_split_ratio))
        test_idx = [i for i in range(len(data)) if i not in train_idx]
        self.data = [data[i] for i in test_idx]
    
    def load_data(self,file_path):
        # data = dda.from_json(file_path)['examples'][0]
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data['examples']
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input = item['input']
        label = item['target']
        return input,label


class GSM8kDataset(Dataset):
    def __init__(self, file_path):
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                question = item['question']
                answer = item['answer']
                target = answer.split('####')[-1].strip()
                data.append({'input': question, 'target': target})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        target = item['target']
        return input_text, target


class AnalyticEntailmentDataset(Dataset):
    def __init__(self, file_path):
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data['examples']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        target_scores = item['target_scores']
        label = 0 if target_scores.get('entailment', 0) == 1 else 1

        return input_text, label



class CivilcommentsDataset(Dataset):
    def __init__(self, file_path):
        data = self.load_data(file_path)
        self.data = data['examples']

    
    def load_data(self,file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input = item['input']
        target_scores = item['toxicity']
        return input, target_scores

class WordUnscramblingDataset(Dataset):
    def __init__(self, file_path):
        data = self.load_data(file_path)
        self.data = data['examples']

    
    def load_data(self,file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input = item['input']
        target = item['target']
        return input, target

class CommonsenseQADataset(Dataset):
    def __init__(self, file_path):
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        target_scores = item['target_scores']
        targets = item['answerKey']
        return input_text, target_scores, targets 
    

class GeneralKnowledgeDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)['examples']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        original_target_scores = item['target_scores']
        new_target_scores = {}
        target = None
        for i, (key, value) in enumerate(original_target_scores.items()):
            option = f'({chr(97+i)})' 
            new_target_scores[option] = key
            if value == 1:
                target = option

        return input_text, new_target_scores, target

class MMluDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['question']
        target_scores = {
            '(a)': item['answer_a'],
            '(b)': item['answer_b'],
            '(c)': item['answer_c'],
            '(d)': item['answer_d']
        }
        target = f"({item['correct_answer'].lower()})"

        return input_text, target_scores, target


class MMluMathDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['question']
        target_scores = {
            '(a)': item['answer_a'],
            '(b)': item['answer_b'],
            '(c)': item['answer_c'],
            '(d)': item['answer_d']
        }
        target = f"({item['correct_answer'].lower()})"

        return input_text, target_scores, target


class TrackingShuffledObjectsDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)['examples']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        original_target_scores = item['target_scores']
        new_target_scores = {}
        target = None
        for i, (key, value) in enumerate(original_target_scores.items()):
            option = f'({chr(97+i)})' 
            new_target_scores[option] = key
            if value == 1:
                target = option

        return input_text, new_target_scores, target


def load_datasets_config(config_path='../model_data_config.json'):
    with open(config_path, 'r') as file:
        data = json.load(file)
    return data['datasets']

def custom_collate_fn(batch):
    if len(batch[0])==2:
        inputs = [item[0] for item in batch]
        target_scores = [item[1] for item in batch]
        return inputs, target_scores
    elif len(batch[0]) == 3:
        inputs = [item[0] for item in batch]
        target_scores = [item[1] for item in batch]
        targets =  [item[2] for item in batch]
        return inputs, target_scores, targets 
    



def load_dataloader(args,dataset_name=None, tokenizer = None, train_split_ratio = 0.6):
    def tokenize_add_label_test(sample):
        input = tokenizer.encode(tokenizer.bos_token + sample["text"], add_special_tokens=False)
        output = tokenizer.encode(sample["labels"], add_special_tokens=False)
        sample = {
            "input_ids": input,
            "attention_mask" : [1] * len(input),
            "labels": output,
            }

        return sample
    
    if dataset_name is None:
        dataset_name = args.dataset_name
    datasets_config = load_datasets_config()
    dataset_path = datasets_config.get(dataset_name, None)
    if dataset_path is None:
        raise ValueError(f"Dataset {dataset_name} not found in configuration.")
    if dataset_name == "empirical_judgments":
        dataset = EmpiricalJudgmentsDataset(dataset_path)
    elif dataset_name == "2_digit_multiplication" or dataset_name == "mod. arithmetic":
        dataset = TwodigitMultiplicationDataset(dataset_path)
    elif dataset_name == "analytic_entailment" or dataset_name == "analytic_entailment_interv":
        dataset = AnalyticEntailmentDataset(dataset_path)
    elif dataset_name == "word_unscrambling":
        dataset = WordUnscramblingDataset(dataset_path)
    elif dataset_name == "civilcomments_maj" or dataset_name == "civilcomments_min" or dataset_name == "civilcomments":
        dataset = CivilcommentsDataset(dataset_path)
    elif dataset_name == "commonsenseqa" or dataset_name == "commonsenseqa_interv": 
        dataset = CommonsenseQADataset(dataset_path)
    elif dataset_name =="general_knowledge":
        dataset = GeneralKnowledgeDataset(dataset_path)
    elif dataset_name =="mmlu":
        dataset = MMluDataset(dataset_path)
    elif dataset_name =="tracking":
        dataset = TrackingShuffledObjectsDataset(dataset_path)
    elif dataset_name =="GSM8k":
        dataset = GSM8kDataset(dataset_path)
    elif dataset_name =="mmlu-math":
        dataset = MMluMathDataset(dataset_path)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported yet.")
    print(f"Data {dataset_name} loaded successfully with Batch size {args.batch_size}.")
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)




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
