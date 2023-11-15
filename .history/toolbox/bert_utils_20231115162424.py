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
            f1 = f1_score(y_true = label, y_pred = predicted_label, average='macro')
            f1_scores.append(f1)
        avg_f1_score = np.mean(f1_scores)
        
        if avg_f1_score > best_f1_score:
            best_f1_score = avg_f1_score
            best_threshold = threshold
    
    return best_threshold, best_f1_score

predictions = np.random.rand(10, 20)
labels = np.random.randint(0, 2, size=(10, 20))
calculate_best_threshold(labels=labels, predictions=predictions)