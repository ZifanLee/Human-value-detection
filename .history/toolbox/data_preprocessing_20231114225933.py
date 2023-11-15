import pandas as pd
import numpy as np
import csv
import torch
from torch.utils.data import DataLoader
from sklearn.utils import class_weight


def create_data_file(path_to_arguments, path_to_labels, output_path=None, drop_duplicates=True):
    """
    Create a data file by merging arguments and labels from the corresponding_files from two files using pandas library.

    Args:
        path_to_arguments (str): File path to the template file.
        path_to_labels (str): File path to the labels file.
        output_path (str, optional): File path to save the output data file.
        drop_duplicates (bool, optional): Whether to drop duplicate rows in the output data file.

    Returns:
        pandas.DataFrame: Merged data with duplicate rows dropped (if drop_duplicates is True).
    """
    
    # Load Arguments & Concat Premise Stance Conclusion
    df_arguments = pd.read_csv(path_to_arguments, sep='\t', quoting=csv.QUOTE_NONE, encoding="utf-8", header=0)
    df_arguments["text"] = df_arguments["Premise"] + " " + df_arguments["Stance"] + " " + df_arguments["Conclusion"]
    
    # Load Labels & Convert to Ints and Create Column with List of Label Names
    df_labels = pd.read_csv(path_to_labels, sep='\t', quoting=csv.QUOTE_NONE, encoding="utf-8", header=0)
    
    for col in df_labels.columns[1:]:
        df_labels[col] = df_labels[col].astype(int)
        
    df_labels['category'] = df_labels.apply(lambda row: [col for col in df_labels.columns if row[col] == 1], axis=1)
    
    # Reorder Columnames
    cols = df_labels.columns.tolist()
    new_cols = cols[:-1]      # All columns except the last one
    new_cols.insert(1, cols[-1])  # Insert the last column to the second position
    df_labels = df_labels[new_cols]
    
    # Merge Data based on ID
    df_merged = pd.merge(df_arguments, df_labels, on=["Argument ID"])

    if drop_duplicates:
        df_merged = df_merged.drop_duplicates(subset=["text"])

    if output_path:
        df_merged.to_csv(output_path)
        
    return df_merged


def calculate_class_weights(targets, label_columns=None):
    if label_columns:
        class_labels = targets[label_columns].values.ravel()
    else:
        class_labels = targets.values.ravel()

    class_labels_tensor = torch.from_numpy(class_labels)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(class_labels_tensor), class_labels_tensor)
    return torch.tensor(class_weights, dtype=torch.float)