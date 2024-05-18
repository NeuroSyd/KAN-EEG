from kan1 import KAN
import torch.nn as nn
import torch.optim as optim
from datasets import EEG_generator, EEG_generator_Time, Epilepsia_12s_STFT, RPA_generator
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, f1_score
from sklearn import metrics
import argparse
import torch

parser = argparse.ArgumentParser(description='Sequential Decision Making..')
parser.add_argument('--dataset', type=str,
                    default='RPAH_Shallow_32_32',
                    help='path to load the model')

parser.add_argument('--load', type=str,
                    default="/home/yikai/Luis_project/KAN-EEG/Results/64B-TUH_STFT_Shallow_Model_32_32/Latest_ckp-64B-TUH_STFT_Shallow_Model_32_32.pth",
                    help='path to load the model')

parser.add_argument('--save', type=str, default='./models/',
                    help='path to load the model')
parser.add_argument('--batch', type=int, default= '64',
                    help="number of branches")
args = parser.parse_args()

torch.manual_seed(42)
batch_size = args.batch

# model = KAN([23 * 125 * 19, 764, 256, 2]) #23 and 125
model = KAN([23 * 125 * 19, 32, 32, 2]) #23 and 125

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# Define loss
criterion = nn.CrossEntropyLoss()
model.to(device)

def plot_AUROC (target_1,output_1,output_file_fol, auroc, pat_name):

    fpr, tpr, thresholds = metrics.roc_curve(target_1, output_1)
    # Create ROC curve plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auroc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    # Save the plot to a file
    output_folder = output_file_fol
    os.makedirs(output_folder, exist_ok=True)

    outputfold = os.path.join(output_folder + str(pat_name))

    with open(outputfold + '.txt', 'a') as file:
        file.write(str(auroc))

    plt.savefig(outputfold)

def calculate_auroc(labels, predicted_probs):

    return roc_auc_score(labels, predicted_probs)

def plot_results(output_file,output_file_fol):

    epochs = []
    valid_acc = []
    val_auroc = []
    val_recall = []
    val_precision = []
    train_loss = []

    # Read the log file and extract metric values

    with open(output_file, 'r') as log_file:
        for line in log_file:
            if line.startswith('epoch'):
                parts = line.strip().split(',')
                epoch = int(parts[0].split(':')[1].strip())
                loss = float(parts[1].split(':')[1].strip())
                acc = float(parts[3].split(':')[1].strip())
                auroc = float(parts[4].split(':')[1].strip())
                recall = float(parts[5].split(':')[1].strip())
                precision = float(parts[6].split(':')[1].strip())

                epochs.append(epoch)
                valid_acc.append(acc)
                val_auroc.append(auroc)
                val_recall.append(recall)
                val_precision.append(precision)
                train_loss.append(loss)

    # Create plots
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, valid_acc, label='Valid Acc')
    plt.plot(epochs, val_auroc, label='Val AUROC')
    plt.plot(epochs, val_recall, label='Val Recall')
    plt.plot(epochs, val_precision, label='Val Precision')
    plt.plot(epochs, train_loss, label="train_loss")

    # Add labels and legend
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics Over Epochs')
    plt.legend()

    outputfold = os.path.join(output_file_fol + str(batch_size) + 'B-' + str(args.dataset) + '.png')

    plt.savefig(outputfold)

model_ckp = torch.load(args.load)

def test_RPA(model):

    folder_path = [
        '/mnt/data7_4T/temp/yikai/RPA_AUC_stft_ICA_totalPat/2011',
        '/mnt/data7_4T/temp/yikai/RPA_AUC_stft_ICA_totalPat/2012',
        '/mnt/data7_4T/temp/yikai/RPA_AUC_stft_ICA_totalPat/2013',
        '/mnt/data7_4T/temp/yikai/RPA_AUC_stft_ICA_totalPat/2014',
        '/mnt/data7_4T/temp/yikai/RPA_AUC_stft_ICA_totalPat/2015',
        '/mnt/data7_4T/temp/yikai/RPA_AUC_stft_ICA_totalPat/2016',
        '/mnt/data7_4T/temp/yikai/RPA_AUC_stft_ICA_totalPat/2017',
        '/mnt/data7_4T/temp/yikai/RPA_AUC_stft_ICA_totalPat/2018',
        '/mnt/data7_4T/temp/yikai/RPA_AUC_stft_ICA_totalPat/2019',
    ]

    for folders1 in folder_path:

        patnames = os.listdir(folders1)
        year = folders1[-4:]  # Extract the year part /RPA
        print(year)

        for patname in patnames:

            print(patname)
            valloader = RPA_generator(patname=str(patname), year = year, batch_size=batch_size)

            # Validation
            model.eval()
            val_loss = 0
            val_accuracy = 0

            predictS = []
            true_labels = []
            predicted1 = []

            output_file_fol = "./Results/" + str(batch_size) + 'B-' + str(args.dataset) + "/" + year + "/"
            os.makedirs(output_file_fol, exist_ok=True)

            with torch.no_grad():
                for images, labels in valloader:
                    images = images.view(-1, 23 * 125 * 19).to(device)
                    output = model(images)
                    labels = labels.cpu()
                    _, predicted = torch.max(output.data, 1)
                    predicted = predicted.cpu()
                    labels = labels.cpu()
                    val_loss += criterion(output, labels.to(device)).item()
                    val_accuracy += ((output.argmax(dim=1) == labels.to(device)).float().mean().item())
                    predicted1.append(predicted)
                    predictS.append(output.softmax(dim=1)[:,1])
                    true_labels.append(labels)

            val_loss /= len(valloader)
            val_accuracy /= len(valloader)

            output_1 = torch.cat(predictS, axis=0)
            target_1 = torch.cat(true_labels, axis=0)

            if sum(target_1.cpu()) != 0:

                val_auroc = calculate_auroc(target_1.cpu(), output_1.cpu())
                print("val_auroc: ", val_auroc)

                plot_AUROC(target_1.cpu(), output_1.cpu(), output_file_fol ,val_auroc, str(patname))

test_RPA(model_ckp)
