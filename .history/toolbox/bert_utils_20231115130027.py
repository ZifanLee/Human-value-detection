import numpy as np
from sklearn.metrics import f1_score

def calculate_best_threshold(labels, predictions):
    thresholds = np.linspace(0, 1, num=100) 
    best_threshold = 0
    best_f1_score = 0
    
    for threshold in thresholds:
        f1_scores = []
        for i in range(labels.shape[0]):
            prediction = predictions[i]
            label = labels[i]
            predicted_label = (prediction > threshold).astype(int)
            f1_score = f1_score(label, predicted_label, average='samples')
            f1_scores.append(f1_score)
        avg_f1_score = np.mean(f1_scores)
        
        if avg_f1_score > best_f1_score:
            best_f1_score = avg_f1_score
            best_threshold = threshold
    
    return best_threshold, best_f1_score

predictions = np.array([0.2, 0.4, 0.6, 0.8])
labels = np.array([0, 1, 1, 0])
print(f1_score(labels=labels, predictions=predictions, average='marco'))