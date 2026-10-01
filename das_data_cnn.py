#coding = UTF-8
import sys


class Logger(object):
    # Duplicates everything printed to stdout into a log file as well.
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger('result.log', sys.stdout)
import argparse
import os
import numpy as np
import time
from sklearn.metrics import accuracy_score, confusion_matrix
from models import CNN
from mydataset import MyDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # used to draw the confusion-matrix heatmap

CLASS_NAMES = ['background', 'digging', 'knocking', 'watering', 'shaking', 'walking']


def test(model, dataset, criterion):
    # Evaluate the model on the test set.
    # Returns: accuracy, mean loss per batch, the 400-dim features of every test sample
    # (with the label appended as the last column), and the confusion matrix.
    model.eval()
    total_batch_num = 0
    val_loss = 0
    prediction = []
    labels = []
    feature_list = []   # one (batch, 401) tensor per batch, concatenated at the end
    with torch.no_grad():   # no gradients needed for evaluation (saves memory and time)
        for (step, i) in enumerate(dataset):
            total_batch_num = total_batch_num+1
            batch_x = i['data']
            batch_y = i['label']
            batch_x = torch.unsqueeze(batch_x, dim=1)  # add channel dim -> (batch, 1, 10000, 12)
            batch_x = batch_x.float()
            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            feature, probs = model(batch_x)   # feature: (batch, 400)
            batch_label = batch_y.unsqueeze(1).float()
            feature_label = torch.cat((feature, batch_label), dim=1)   # (batch, 401)
            feature_list.append(feature_label.cpu())
            loss = criterion(probs, batch_y)
            _, pred = torch.max(probs, dim=1)   # predicted class = highest score
            prediction.extend(pred.tolist())
            labels.extend(batch_y.tolist())
            val_loss += loss.item()
    feature_list = torch.cat(feature_list, dim=0)   # (n_test_samples, 401)
    accuracy = accuracy_score(labels, prediction)
    C = confusion_matrix(labels, prediction, labels=list(range(len(CLASS_NAMES))))
    return accuracy, val_loss/total_batch_num, feature_list, C


def train(model, train_x, train_y, optimizer, criterion):
    # One optimisation step on a single batch.
    model.train()
    model.zero_grad()
    _, probs = model(train_x)
    loss = criterion(probs, train_y)
    _, pred = torch.max(probs, dim=1)
    labels = train_y.tolist()
    predi = pred.tolist()
    loss.backward()
    optimizer.step()
    return labels, predi, loss.item()


def draw(train_acc, train_loss, test_acc, test_loss):
    # Plot the training/test accuracy and loss curves over the epochs.
    epochs = range(len(train_acc))
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_acc, 'o-', label="train", color='b')
    plt.plot(epochs, test_acc, 'o-', label="test", color='r')
    plt.legend(loc='upper left')
    plt.title('accuracy & loss vs. epochs')
    plt.ylabel('accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_loss, '.-', label="train", color='b')
    plt.plot(epochs, test_loss, '.-', label="test", color='r')
    plt.legend(loc='upper left')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig("accuracy_loss.jpg")
    plt.show()


def safe_div(a, b):
    # Division that returns 0 instead of raising / producing NaN when b == 0.
    return a / b if b != 0 else 0.0


def draw_result(C):
    # Draw the confusion matrix and print the evaluation metrics.
    # C[i][j] = number of samples whose true class is i and predicted class is j.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    df = pd.DataFrame(C)
    sns.heatmap(df, fmt='g', annot=True, robust=True,
                annot_kws={'size': 10},
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted label', fontsize=15)  # x axis
    ax.set_ylabel('True label', fontsize=15)  # y axis
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    plt.savefig('./CNN_confusion_matrix.jpg')
    plt.show()

    # Accuracy: correctly classified samples (the diagonal) / all samples
    Acc = safe_div(np.trace(C), np.sum(C))
    print('acc: %.4f' % Acc)
    # NAR (nuisance alarm rate): background samples predicted as an event (false alarms)
    # divided by all samples predicted as an event (all alarms). See README.
    NAR = safe_div(np.sum(C[0, 1:]), np.sum(C[:, 1:]))
    print('NAR: %.4f' % NAR)
    # FNR (false negative rate): real events predicted as background (missed events)
    # divided by all real events.
    FNR = safe_div(np.sum(C[1:, 0]), np.sum(C[1:, :]))
    print('FNR: %.4f' % FNR)
    column_sum = np.sum(C, axis=0)   # number of predictions per class
    print(column_sum)
    row_sum = np.sum(C, axis=1)      # number of true samples per class
    print(row_sum)

    # Per-class precision, recall and F1-score (classes are numbered 1-6 in the output).
    for i in range(1, len(C) + 1):
        Precision = safe_div(C[i - 1][i - 1], column_sum[i - 1])
        Recall = safe_div(C[i - 1][i - 1], row_sum[i - 1])
        F1 = safe_div(2*Precision*Recall, Precision+Recall)
        print('precision_%d: %.3f' % (i, Precision))
        print('Recall_%d: %.3f' % (i, Recall))
        print('F1_%d: %.3f' % (i, F1))


def main(args):

    # Load the training and test sets
    train_dataset = MyDataset(args.root, args.txtpath, transform=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = MyDataset(args.root2, args.txtpath2, transform=None)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    models = {"CNN": CNN}
    model = models[args.model]()
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    train_time = 0
    for epoch in range(args.epochs):
        tic = time.time()
        train_predict = []
        train_label = []
        runloss = 0
        for (cnt, i) in enumerate(train_loader):
            batch_x = i['data']
            batch_y = i['label']
            batch_x = torch.unsqueeze(batch_x, dim=1)   # (batch, 1, 10000, 12)
            batch_x = batch_x.float()
            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            tlabels, tpredi, tloss = train(model, batch_x, batch_y, optimizer, criterion)
            runloss = runloss+tloss
            train_label.extend(tlabels)
            train_predict.extend(tpredi)
        per_epoch_train_time = time.time()-tic
        train_time = per_epoch_train_time+train_time
        taccuracy = accuracy_score(train_label, train_predict)
        train_acc_list.append(taccuracy)
        loss = runloss / len(train_loader)   # mean loss per batch (includes the last partial batch)
        train_loss_list.append(loss)
        acc_score, loss_score, feature_list, C = test(model, test_loader, criterion)
        print("Epoch %d Train_accuracy %.3f Train_loss %.3f Val_accuracy %.3f Val_loss %.3f"
              % (epoch, taccuracy, loss, acc_score, loss_score))
        test_acc_list.append(acc_score)
        test_loss_list.append(loss_score)
        if epoch == args.epochs-1:  # last epoch: save the model, features and results
            torch.save(model, args.save)
            feature_list = feature_list.numpy()
            np.savetxt('feature_data.csv', feature_list, delimiter=',')  # save the feature vectors (used by feature_visualization.py)
            draw_result(C)
            print('train totally using %.3f seconds' % train_time)
    draw(train_acc_list, train_loss_list, test_acc_list, test_loss_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CNN for classification")

    '''save model'''
    parser.add_argument("--save", type=str, default="model.pth",
                        help="path to save model")
    '''model parameters'''
    # By default the dataset is expected in a "das_data" folder next to this script.
    rootpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'das_data')
    parser.add_argument("--root", type=str, default=rootpath + '/train',
                        help="rootpath of traindata")
    parser.add_argument("--root2", type=str, default=rootpath + '/test',
                        help="rootpath of valdata")
    parser.add_argument("--txtpath", type=str, default=rootpath + '/train/label.txt',
                        help="path of train_list")
    parser.add_argument("--txtpath2", type=str, default=rootpath + '/test/label.txt',
                        help="path of val_list")
    parser.add_argument("--model", type=str, default="CNN",
                        help="type of model to use for classification")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="batch size")
    my_args = parser.parse_args()

    main(my_args)
