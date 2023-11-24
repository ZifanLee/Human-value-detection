import csv

def extract_metrics_from_md(file_path):
    train_loss = []
    valid_loss = []
    f1_score = []
    precision = []
    recall = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if 'Train Loss:' in line:
            train_loss.append(float(line.split('Train Loss:')[1].split('-')[0].strip()))
            valid_loss.append(float(line.split('Val Loss:')[1].split('-')[0].strip()))
        elif 'F1 Score:' in line:
            f1_score.append(float(line.split('F1 Score:')[1].split('-')[0].strip()))
            precision.append(float(line.split('Precision:')[1].split('-')[0].strip()))
            recall.append(float(line.split('Recall:')[1].strip()))

    return train_loss, valid_loss, f1_score, precision, recall


def write_metrics_to_csv(train_loss, valid_loss, f1_score, precision, recall):
    data = zip(train_loss, valid_loss, f1_score, precision, recall)

    with open("./loss.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Train Loss', 'Valid Loss', 'F1 Score', 'Precision', 'Recall'])
        writer.writerows(data)
        
    print('Data written to CSV successfully!')
    
write_metrics_to_csv(extract_metrics_from_md("/Users/lizifan/Desktop/nlp project/nlp data.md"))