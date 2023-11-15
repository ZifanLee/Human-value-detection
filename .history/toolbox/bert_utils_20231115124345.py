import numpy as np
from sklearn.metrics import f1_score

def calculate_best_threshold(labels, predictions):
    thresholds = np.linspace(0, 1, num=100) 
    best_threshold = 0
    best_f1_score = 0
    
    for threshold in thresholds:
        predicted_labels = (predictions > threshold).astype(int)
        f1_scores = f1_score(labels, predicted_labels, average='samples')
        avg_f1_score = np.mean(f1_scores)
        
        if avg_f1_score > best_f1_score:
            best_f1_score = avg_f1_score
            best_threshold = threshold
    
    return best_threshold, best_f1_score

