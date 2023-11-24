import csv
import pandas as pd
import matplotlib.pyplot as plt

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
    
write_metrics_to_csv(*extract_metrics_from_md("/Users/lizifan/Desktop/nlp project/nlp data.md"))


def draw_line_chart_from_csv(file_path, column_name):
    # Read the data from CSV file using pandas
    df = pd.read_csv(file_path)

    # Extract the specified column data
    column_data = df[column_name]

    # Create x-axis values (epochs)
    epochs = range(1, len(column_data) + 1)

    # Plot the line chart
    plt.plot(epochs, column_data)

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel(column_name)
    plt.title(f'{column_name} over Epochs')

    # Display the chart
    plt.show()

draw_line_chart_from_csv('./loss.csv', 'F1 Score')