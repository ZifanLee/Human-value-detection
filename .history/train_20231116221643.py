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


pred  = [0.4956, 0.4604, 0.4621, 0.4840, 0.5068, 0.4802, 0.5011, 0.4889, 0.4610,
         0.5280, 0.4664, 0.4447, 0.4674, 0.5065, 0.4697, 0.4811, 0.4756, 0.5187,
         0.5125, 0.4832]
labels = [1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
         1., 0.]

def calculate_loss(predictions, labels, class_weights=None):
    criterion = torch.nn.BCEWithLogitsLoss(weight=class_weights)
    loss = criterion(predictions, labels)
    print(predictions, labels, loss)
    return loss

calculate_loss(predictions=pred, labels=labels)