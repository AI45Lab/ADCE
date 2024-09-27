from tqdm import tqdm
import logging
import numpy as np
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import re
import json
import numpy as np
import re

def setup_logger(log_file, logger_name='my_logger'):
    logger = logging.getLogger(logger_name)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)  
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

def create_and_get_logger(folder_path):
    log_filename = os.path.join(folder_path, "running.log")
    logger = setup_logger(log_filename)
    return logger

def visualize_layer(p_values,base_path, all_labels,layer_type = 'acts',mode = 'PCA',n_components=2,num_head=0):
    pattern = r"results_[a-zA-Z]/([^_]+)"
    match = re.search(pattern, base_path)
    if match:
        extracted_content = match.group(1)
    else:
        print("No model name found.")
    with open('./model_data_config.json', 'r') as f:
        models = json.load(f)
    with open(models['models'][extracted_content]+'/config.json', 'r') as f:
        config = json.load(f)
    total_layer = config['num_hidden_layers']
    fig, axes = plt.subplots(1, len(p_values), figsize=(8 * len(p_values), 6))

    for i, p in enumerate(tqdm(p_values, desc="Processing layers")):
        layer = int(total_layer * p)
        file_name = f"{base_path}/layer_{layer}_{layer_type}.npy"
        if os.path.isfile(file_name):
            acts = np.load(file_name)
            # print(f"Load layer_{layer}_acts successfully!")
        else:
            print(f"Load layer_{layer}_acts fail!")
            continue
        # print(acts.shape)
        if layer_type == 'acts' and acts.shape[0] != 1:
            shape = acts.shape
            acts = acts.reshape(1, shape[0] * shape[1], shape[2])
        elif layer_type == 'attns' and acts.shape[0] != 1:
            acts = acts[:, num_head:num_head+1, :, :].transpose(1, 0, 2, 3)
            shape = acts.shape
            acts = acts.reshape(1, shape[1], shape[2]* shape[3])
        data = acts[0]
        if mode =='PCA':
            reduce_dimention = PCA(n_components=n_components)
        elif mode == 'UMAP':
            reduce_dimention = umap.UMAP(n_components=n_components)
        principal_components_2 = reduce_dimention.fit_transform(data)
        principal_components_2 = principal_components_2[:, :2]

        ax = axes[i]
        for label in np.unique(all_labels):
            ax.scatter(principal_components_2[np.array(all_labels) == label, 0],
                       principal_components_2[np.array(all_labels) == label, 1],
                       label=label)
        ax.set_title(f'{mode} of {extracted_content}={layer_type} with ({p*100}% depth )')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.legend()

    plt.tight_layout()
    plt.show()


def process_predictions(file_path, group_size=4):
    predictions = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            pred_match = re.search(r'prediction: (None|-?\d+)', line)
            label_match = re.search(r'label: (-?\d+)', line)
            if pred_match and label_match:
                prediction = pred_match.group(1)
                if prediction == 'None':
                    prediction = -2
                else:
                    prediction = int(prediction)
                label = int(label_match.group(1))
                predictions.append(prediction)
                labels.append(label)
    pre_trans = []
    label_trans = []
    for i in range(0, len(predictions), group_size):
        pred_group = predictions[i:i+group_size]
        label_group = labels[i:i+group_size]
        true_numbers = list(set([label for label in label_group if label != -1]))
        label_trans_group = [1 if label == -1 else 0 for label in label_group]
        pre_trans_group = []
        for j in range(group_size):
            if j < group_size // 2:  
                pre_trans_group.append(0 if pred_group[j] == label_group[j] else 1)
            else:  
                pre_trans_group.append(0 if pred_group[j] in true_numbers else 1)
        pre_trans.extend(pre_trans_group)
        label_trans.extend(label_trans_group)
    return pre_trans, label_trans