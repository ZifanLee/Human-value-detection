import torch
import pytorch_lightning as pl

import numpy as np

from transformers import (
    get_linear_schedule_with_warmup,
    AutoModel
)
import torch.nn as nn
import torch.functional as F
from torchmetrics import AUROC, F1Score
from torch.optim import AdamW

from sklearn.metrics import classification_report
from toolbox.bert_utils import max_for_thres


class BertFineTunerPl(pl.LightningModule):

    def __init__(self, n_classes: int, params, label_columns, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.bert = AutoModel.from_pretrained(params["MODEL_PATH"], return_dict=True)
        self.hidden_layers = nn.ModuleList()
        output_dim = self.bert.config.hidden_size
        
        self.hidden_layers.append(nn.Linear(output_dim,256))
        self.hidden_layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.hidden_layers.append(nn.Dropout(0.1))
        
        self.hidden_layers.append(nn.Linear(256,256))
        self.hidden_layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.hidden_layers.append(nn.Dropout(0.1))
        
        self.hidden_layers.append(nn.Linear(256,180))
        self.hidden_layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.hidden_layers.append(nn.Linear(180,128))
        self.hidden_layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.hidden_layers.append(nn.Linear(128,128))
        self.hidden_layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.hidden_layers.append(nn.Linear(128,64))
        self.hidden_layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.hidden_layers.append(nn.Linear(64,32))
        self.hidden_layers.append(nn.LeakyReLU(negative_slope=0.2))
        
        
        self.classifier = nn.Linear(32,20)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCEWithLogitsLoss()
        self.params = params
        self.label_columns = label_columns
        

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        for layer in self.hidden_layers:
            output = layer(output)
        output_cls = self.classifier(output)
        output = torch.sigmoid(output_cls)

        loss = 0
        if labels is not None:
            for i in range(len(self.label_columns)):
                loss += self.criterion(output_cls[:, i], labels[:, i])
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.training_step_outputs.append([loss, outputs, labels])
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.validation_step_outputs.append([loss, outputs, labels])  
        return {"val_loss": loss, "predictions": outputs, "labels": labels}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"test_loss": loss, "predictions": outputs, "labels": labels}

    def on_validation_epoch_end(self):
        losses = [x[0] for x in self.validation_step_outputs]
        outputs = [x[1] for x in self.validation_step_outputs]
        labels = [x[2] for x in self.validation_step_outputs]
        avg_loss = torch.stack(losses).mean()

        labels = []
        predictions = []
        for output in self.validation_step_outputs:
            out_labels = output[2].detach().cpu()
            out_predictions = output[1].detach().cpu()
            labels.append(out_labels)
            predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        # Calculate accuracy for each class
        for i, name in enumerate(self.label_columns):
            class_predictions = predictions[:, i]
            class_labels = labels[:, i]
            class_accuracy = self.calculate_accuracy(class_predictions, class_labels)
            self.log(f"{name}_accuracy/Val", class_accuracy, logger=True)

        self.log("avg_val_loss", avg_loss, prog_bar=True, logger=True)

        y_pred = predictions.numpy()
        y_true = labels.numpy()

        # Calculate other evaluation metrics for each class
        for i, name in enumerate(self.label_columns):
            class_y_pred = y_pred[:, i]
            class_y_true = y_true[:, i]
            class_f1_score = self.calculate_f1_score(class_y_pred, class_y_true)
            class_precision = self.calculate_precision(class_y_pred, class_y_true)
            class_recall = self.calculate_recall(class_y_pred, class_y_true)
            self.log(f"{name}_f1-score/Val", class_f1_score, logger=True)
            self.log(f"{name}_precision/Val", class_precision, logger=True)
            self.log(f"{name}_recall/Val", class_recall, logger=True)
        
        return {"avg_val_loss": avg_loss}
    
    
    def on_train_epoch_end(self):
        losses = [x[0] for x in self.training_step_outputs]
        outputs = [x[1] for x in self.training_step_outputs]
        avg_loss = torch.stack(losses).mean()

        labels = []
        predictions = []
        for output in self.training_step_outputs:
            out_labels = output[2].detach().cpu()
            out_predictions = output[1]["predictions"].detach().cpu()
            labels.append(out_labels)
            predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        # Calculate accuracy for each class
        for i, name in enumerate(self.label_columns):
            class_predictions = predictions[:, i]
            class_labels = labels[:, i]
            class_accuracy = self.calculate_accuracy(class_predictions, class_labels)
            self.log(f"{name}_accuracy/Train", class_accuracy, logger=True)

        self.log("avg_train_loss", avg_loss, logger=True)
    

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.params['LR'])

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

    def _cls_embeddings(self, output):
        '''Returns the embeddings corresponding to the <CLS> token of each text. '''

        last_hidden_state = output[0]
        cls_embeddings = last_hidden_state[:, 0]
        return cls_embeddings

    def _meanPooling(self, output, attention_mask):
        '''Performs the mean pooling operation. '''

        last_hidden_state = output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def _maxPooling(self, output, attention_mask):
        '''Performs the max pooling operation. '''

        last_hidden_state = output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        last_hidden_state[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        max_embeddings = torch.max(last_hidden_state, 1)[0]
        return max_embeddings
