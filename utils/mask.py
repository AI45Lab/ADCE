import torch
import re
import random
from torch.utils.data import Dataset

class MaskedWordDataset(Dataset):
    def __init__(self, texts, labels, flags, num_to_mask, tokenizer, interv_type, args):
        self.texts = texts
        self.scores = labels
        self.true_labels = labels
        self.labels = labels
        self.flags = flags
        self.num_to_mask = num_to_mask
        self.tokenizer = tokenizer
        self.interv_type = interv_type
        self.args = args
        self.mask_fix_position = args.mask_fix_position
        self.additional_mask_words = set(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                                          'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
                                          'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million', 'billion',
                                          'times', 'minus', 'plus', 'divided', 'multiplied', 'dozen', 'once', 'twice'])

    def is_maskable(self, word):
        return word.isdigit() or word.lower() in self.additional_mask_words

    def mask_word(self, text, labels):
        words = re.findall(r'\w+|[^\w\s]', text)
        
        maskable_positions = [i for i, word in enumerate(words) if self.is_maskable(word)]
        if not maskable_positions and self.mask_fix_position == 0:
            return text, labels, None, None  # Skip this sentence if no maskable words
        if self.mask_fix_position != 0:
            if len(words) > self.mask_fix_position:
                mask_position = self.mask_fix_position
            else:
                return text, labels, None, None  # Skip if fixed position is not maskable
        else:
            mask_position = random.choice(maskable_positions)

        original_word = words[mask_position]

        if self.interv_type == 'mask':
            words[mask_position] = '<mask>'
        elif self.interv_type == 'replace':
            vocab = list(self.tokenizer.get_vocab().keys())
            words[mask_position] = random.choice(vocab)
        masked_text = " ".join(words)
        new_labels = torch.tensor([-2])  
        return masked_text, new_labels, mask_position, original_word

    def mask_context(self, text, labels, mask_position, original_word):
        words = re.findall(r'\w+|[^\w\s]', text)
        words[mask_position] = original_word  
        context_start = max(0, mask_position - 3)
        context_end = min(len(words), mask_position + 4)
        context = words[context_start:context_end]
        
        non_maskable_positions = [i for i, word in enumerate(context) 
                                  if not self.is_maskable(word) and i + context_start != mask_position]
        if not non_maskable_positions:
            return text, labels  

        non_maskable_mask_position = random.choice(non_maskable_positions)
        actual_non_maskable_position = context_start + non_maskable_mask_position

        if self.interv_type == 'mask':
            words[actual_non_maskable_position] = '<mask>'
        elif self.interv_type == 'replace':
            vocab = list(self.tokenizer.get_vocab().keys())
            words[actual_non_maskable_position] = random.choice(vocab)
        elif self.interv_type == 'swapping':
            words[mask_position], words[actual_non_maskable_position] = words[actual_non_maskable_position], words[mask_position]

        masked_text = " ".join(words)
        return masked_text, labels  
        
    def create_masked_dataset(self):
        if self.args.prompt == 'csuperhigh' or self.args.prompt == 'fewshot_csuperhigh':
            for key, value in self.true_labels.items():
                if value.item() == 1.0:
                    flags = str(self.flags) + '_' + str(key)
                    break
        elif self.args.prompt == 'superhigh' or self.args.prompt == 'fewshot_superhigh':
            flags = str(self.flags) + '_' + str(self.true_labels)

        original_texts_and_labels = []
        context_masked_texts_and_labels = []
        word_masked_texts_and_labels = []

        for text in self.texts:
            original_texts_and_labels.append((text, self.labels, self.true_labels))
            
            for _ in range(self.num_to_mask):
                word_masked_text, word_labels, mask_position, original_word = self.mask_word(text, self.labels)
                if mask_position is not None:
                    
                    context_masked_text, context_labels = self.mask_context(text, self.labels, mask_position, original_word)
                    context_masked_texts_and_labels.append((context_masked_text, context_labels, self.true_labels))
                    
                   
                    word_masked_texts_and_labels.append((word_masked_text, word_labels, self.true_labels))

       
        all_texts_and_labels = original_texts_and_labels + context_masked_texts_and_labels + word_masked_texts_and_labels

        masked_texts = [item[0] for item in all_texts_and_labels]
        if self.args.prompt == 'superhigh' or self.args.prompt == 'fewshot_superhigh':
            labels = [item[1][0] for item in all_texts_and_labels]
        elif self.args.prompt == 'csuperhigh' or self.args.prompt == 'fewshot_csuperhigh':
            labels = [item[1] for item in all_texts_and_labels]

        data_flags = []
        count = 0
        for item in all_texts_and_labels:
            data_flag = 'position' + flags + '_' + str(count) + ' ' + str(item[0])
            data_flags.append(data_flag)
            count += 1

        return masked_texts, labels, data_flags
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], self.flags[idx]

def collate_fn(batch, tokenizer):
    texts, labels, flags = zip(*batch)
    texts = [text for text in texts]
    labels = [label for label in labels]
    flags = [flag for flag in flags]
    return texts, labels, flags
