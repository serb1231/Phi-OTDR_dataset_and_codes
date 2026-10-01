#coding = UTF-8
import copy
import sys
import torch
from tqdm import tqdm
import logging.config

NUM_EPOCHS = 10
LOCAL_ITERS = 2
VIS_DATA = False
BATCH_SIZE = 5
NUM_CLIENTS = 2
DATASET = "Fiber_Optic_Detection"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)

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
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

CLASS_NAMES = ['background', 'digging', 'knocking', 'watering', 'shaking', 'walking']


def FedAvg(params):
    """
    Average the paramters from each client to update the global model
    :param params: list of paramters from each client's model
    :return global_params: average of paramters from each client
    """
    global_params = copy.deepcopy(params[0])
    for key in global_params.keys():
        for param in params[1:]:
            global_params[key] += param[key]
        global_params[key] = torch.div(global_params[key], len(params))
    return global_params

def test(model, dataloader, device):
    """
    Tests the Federated global model for the given dataset
    :param model: Trained CNN model for testing
    :param dataloader: data iterator used to test the model
    :return test_loss: test loss for the given dataset
    :return preds: predictions for the given dataset
    :return accuracy: accuracy for the prediction values from the model
    """
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    all_preds = []
    model.eval()
    with torch.no_grad():   # no gradients needed for evaluation
        for batch in tqdm(dataloader, total=len(dataloader)):
            data = batch['data'].unsqueeze(1).float().div(255.0)  # (B, 10000, 12) int -> (B, 1, 10000, 12) float
            target = batch['label']

            data, target = data.to(device), target.to(device)
            _, output = model(data)   # the CNN returns (feature, class scores)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)   # loss summed over the samples of the batch
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            all_preds.append(preds.cpu())
    accuracy = correct / len(dataloader.dataset)

    return test_loss / len(dataloader.dataset), torch.cat(all_preds), accuracy

def train(local_model, device, dataloader, iters, lr):
    """
    Trains a local model for a given client
    :param local_model: a copy of global CNN model required for training
    :param device: the device used to train the model - GPU/CPU
    :param dataloader: DataLoader over this client's share of the training data
    :return local_params: parameters from the trained model from the client
    :return train_loss: training loss for the current epoch
    """
    #optimzer for training the local models
    local_model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
    train_loss = 0.0
    local_model.train()
    #Iterate for the given number of Client Iterations
    for i in range(iters):
        batch_loss = 0.0
        for batch in tqdm(dataloader, total=len(dataloader)):
            # Each batch is a dict {'data': (B, 10000, 12), 'label': (B,)}
            data = batch['data'].unsqueeze(1).float().div(255.0).to(device)   # (B, 1, 10000, 12) float
            target = batch['label'].to(device)
            #set gradients to zero
            optimizer.zero_grad()
            #Get output prediction from the Client model
            _, output = local_model(data)
            #Computer loss
            loss = criterion(output, target)
            batch_loss += loss.item()*data.size(0)
            #Collect new set of gradients
            loss.backward()
            #Update local model
            optimizer.step()
        #add loss for each iteration (batch_loss is summed per sample, so divide by the number of samples)
        train_loss+=batch_loss/len(dataloader.dataset)
    return local_model.state_dict(), train_loss/iters

def main(args):
    if not os.path.isdir('models'):
        os.mkdir('models')
    if not os.path.isdir('results'):
        os.mkdir('results')

    # Initialize a logger to log epoch results
    logname = ('results/log_federated_' + DATASET + "_" + str(NUM_EPOCHS) + "_" + str(NUM_CLIENTS) + "_" + str(
        LOCAL_ITERS))
    logging.basicConfig(filename=logname, level=logging.DEBUG)
    logger = logging.getLogger()

    # Load the training and test sets
    train_dataset = MyDataset(args.root, args.txtpath, transform=None)
    # Distribute the training data across clients.
    # label.txt is sorted by class, so shuffle the indices first; otherwise each client
    # would only get a few of the 6 event types.
    indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(0)).tolist()
    client_loaders = []
    for client in range(NUM_CLIENTS):
        client_subset = Subset(train_dataset, indices[client::NUM_CLIENTS])
        client_loaders.append(DataLoader(client_subset, batch_size=args.batch_size, shuffle=True))
        print('Client %d: %d training samples' % (client, len(client_subset)))


    test_dataset = MyDataset(args.root2, args.txtpath2, transform=None)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    models = {"CNN_FEDERATED": CNN}
    global_model = models[args.model]().to(DEVICE)
    global_params = global_model.state_dict()

    global_model.train()
    all_train_loss = list()
    all_val_loss = list()
    val_loss_min = np.inf


    for epoch in range(args.epochs):
        print("\nEpoch :", str(epoch))
        local_params, local_losses = [], []
        # Send a copy of global model to each client
        for idx in range(NUM_CLIENTS):
            # Perform training on client side and get the parameters
            param, loss = train(copy.deepcopy(global_model), DEVICE, client_loaders[idx], LOCAL_ITERS, args.lr)
            local_params.append(copy.deepcopy(param))
            local_losses.append(copy.deepcopy(loss))

        # Federated Average for the paramters from each client
        global_params = FedAvg(local_params)
        # Update the global model
        global_model.load_state_dict(global_params)
        all_train_loss.append(sum(local_losses) / len(local_losses))

        # Test the global model
        val_loss, _, accuracy = test(global_model, test_loader, device=DEVICE)
        all_val_loss.append(val_loss)

        epoch_summary = 'Epoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}' \
            .format(epoch, args.epochs, all_train_loss[-1], val_loss, accuracy)
        logger.info(epoch_summary)
        print(epoch_summary)

        # if validation loss decreases, save the model
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            logger.info("Saving Model State")
            torch.save(global_model.state_dict(), "models/" + DATASET + "_" + str(NUM_CLIENTS) + "_federated.sav")


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
    parser.add_argument("--model", type=str, default="CNN_FEDERATED",
                        help="type of model to use for classification")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="batch size")
    my_args = parser.parse_args()

    main(my_args)
