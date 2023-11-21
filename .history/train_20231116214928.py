import pandas as pd
import numpy as np

from tqdm.auto import tqdm

import torch
import torch.nn as nn

from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import pickle

from model.BertDataModule import BertDataModule, BertDataset
from model.BertFineTuner import BertFineTuner, train


RANDOM_SEED = 99
torch.manual_seed(99)

PARAMS = {
    # Language Model and Hyperparameters
    "MODEL_PATH": 'roberta-base',
    "BATCH_SIZE": 32,
    "ACCUMULATE_GRAD_BATCHES": 1,
    "LR": 1e-5,
    "EPOCHS": 3,
    "OPTIMIZER": 'AdamW',
    "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "NUM_TRAIN_WORKERS": 4,
    "NUM_VAL_WORKERS": 4,
    "MAX_TOKEN_COUNT":128,
    "RANDOM_SEED": RANDOM_SEED,

    # Early Stopping Params
    "PATIENCE": 3,
    "VAL_CHECK_INTERVAL": 300,
    
    # The metric we optimize for. Alternative "custom_f1/Val" and "max"
    "MAX_THRESHOLD_METRIC": "custom", #The f1-score that should maximized (custom = formula for the task evaluation)
    "EARLY_STOPPING_METRIC": "avg_val_loss",
    "EARLY_STOPPING_MODE": "min",

    # DATA
    "VALIDATION_SET_SIZE":500,

    "TRAIN_PATH" : "./data/data_training_full.csv", #
    "LEAVE_OUT_DATA_PATH": "./data/leave_out_dataset_300.csv",
    "SAVE_PATH": "./model/best_model.pt"

}

train_df = pd.read_csv(PARAMS["TRAIN_PATH"], index_col=0)
LABEL_COLUMNS = train_df.columns.tolist()[6:]
leave_out_df = pd.read_csv(PARAMS["LEAVE_OUT_DATA_PATH"], index_col=0)

steps_per_epoch=len(train_df) // PARAMS['BATCH_SIZE']
total_training_steps = steps_per_epoch * PARAMS['EPOCHS']
warmup_steps = total_training_steps // 5
warmup_steps, total_training_steps

train_df, val_df = train_test_split(train_df, test_size=PARAMS["VALIDATION_SET_SIZE"], random_state=PARAMS["RANDOM_SEED"])
TOKENIZER = AutoTokenizer.from_pretrained(PARAMS["MODEL_PATH"])

data_module = BertDataModule(
    train_df=train_df,
    val_df=val_df,
    tokenizer=TOKENIZER,
    params=PARAMS,
    label_columns=LABEL_COLUMNS
)

data_module.setup()
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

from sklearn.utils import class_weight

def calculate_class_weights(no_of_classes, samples_per_cls, power=1):
    weights_for_samples = 1.0/np.array(np.power(samples_per_cls,power))
    weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * no_of_classes
    return weights_for_samples

temp_df = pd.concat([train_df, val_df], ignore_index=True)
class_labels = temp_df[LABEL_COLUMNS]
num_ones = class_labels.eq(1).sum()
class_weights = calculate_class_weights(20,num_ones)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(class_weights)

model = BertFineTuner(params=PARAMS, label_columns=LABEL_COLUMNS, n_training_steps=total_training_steps, n_warmup_steps=warmup_steps)

train(
    model = model,
    train_loader = train_loader, 
    val_loader = val_loader, 
    num_epochs = PARAMS['EPOCHS'], 
    learning_rate = PARAMS['LR'], 
    n_warmup_steps = warmup_steps, 
    n_training_steps = total_training_steps,
    save_path = PARAMS['SAVE_PATH'],
    class_weights= class_weights_tensor.to(PARAMS['DEVICE'])
)

trained_weights = torch.load(PARAMS['SAVE_PATH'])
model.load_state_dict(trained_weights)

model.eval()
for param in model.parameters():
    param.requires_grad = False