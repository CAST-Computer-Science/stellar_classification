import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
from utils import set_seeds, predict_by_max_logit, list_errors
from plot import plot_confusion_matrix
from create_dataset import create_dataset
from label_maps import coarse_label_map
from model import MlpModel
from argparse import ArgumentParser
import pandas as pd


parser = ArgumentParser()
parser.add_argument('--test_set', type=str, default='split_train', choices=['split_train', 'galactic_plane', 'bulge'],
                    help='Which test set to use.')
parser.add_argument('--fits_dir', type=str, default='./fits_files',
                    help='Path to directory that contains fits files.')
parser.add_argument('--aors_dir', type=str, default="./aor_files/",
                    help='Path to directory that contains AOR csv files.')
parser.add_argument('--training_dir', type=str, default='./training_data/',
                    help='Path to directory that contains created training data.')
parser.add_argument('--test_dir', type=str, default="./test_data/",
                    help='Path to directory that contains created test data.')
parser.add_argument('--verbose', type=str, default='off', choices=['on', 'off'],
                    help='If verbose is on print out debugging information.')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

num_classes = len(coarse_label_map)
hidden_size_1 = 100
hidden_size_2 = 100
set_seeds(seed=args.seed)

create_dataset(args=args)

if args.test_set == 'split_train':
    features = np.genfromtxt(os.path.join(args.training_dir, "train_features.csv"), delimiter=',', skip_header=0)
    labels = np.genfromtxt(os.path.join(args.training_dir, "train_labels.csv"),
                           delimiter=',', usecols=(2), skip_header=1, dtype=np.dtype(int))
    aors = np.genfromtxt(os.path.join(args.training_dir, "train_labels.csv"),
                         delimiter=',', usecols=(0), skip_header=1, dtype=np.dtype(str))
    aor_indices = np.genfromtxt(os.path.join(args.training_dir, "train_labels.csv"),
                                delimiter=',', usecols=(1), skip_header=1, dtype=np.dtype(str))

    features_train, features_test, labels_train, labels_test, aors_train, aors_test, aor_indices_train, aor_indices_test =\
        train_test_split(features, labels, aors, aor_indices, test_size=0.2, random_state=args.seed, stratify=labels)
else:
    # read the features and labels from the dataset file
    features_train = np.genfromtxt(os.path.join(args.training_dir, "train_features.csv"),
                                   delimiter=',', skip_header=0)
    labels_train = np.genfromtxt(os.path.join(args.training_dir, "train_labels.csv"),
                                 delimiter=',', usecols=(2), skip_header=1, dtype=np.dtype(int))
    aors_train = np.genfromtxt(os.path.join(args.training_dir, "train_labels.csv"),
                               delimiter=',', usecols=(0), skip_header=1, dtype=np.dtype(str))
    aor_indices_train = np.genfromtxt(os.path.join(args.training_dir, "train_labels.csv"),
                                      delimiter=',', usecols=(1), skip_header=1, dtype=np.dtype(str))

    features_test = np.genfromtxt(os.path.join(args.test_dir, "test_features.csv"),
                                  delimiter=',', skip_header=0)
    labels_test = np.genfromtxt(os.path.join(args.test_dir, "test_labels.csv"),
                                delimiter=',', usecols=(2), skip_header=1, dtype=np.dtype(int))
    aors_test = np.genfromtxt(os.path.join(args.test_dir, "test_labels.csv"),
                              delimiter=',', usecols=(0), skip_header=1, dtype=np.dtype(str))
    aor_indices_test = np.genfromtxt(os.path.join(args.test_dir, "test_labels.csv"),
                                     delimiter=',', usecols=(1), skip_header=1, dtype=np.dtype(str))


# convert the numpy arrays to PyTorch tensors
features_train = torch.FloatTensor(features_train)
features_test = torch.FloatTensor(features_test)
labels_train = torch.LongTensor(labels_train)
labels_test = torch.LongTensor(labels_test)

# create the model
model = MlpModel(num_classes=num_classes, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2)

# define a loss function for classification
loss_fn = nn.CrossEntropyLoss()

# use the Adam optimizer to update the weights
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 1000  # one epoch is reading through the dataset a single time
losses = []  # a list to hold the loss values after each epoch

# train the model
for i in range(epochs):
    logits = model(features_train)  # pass the features forward through the model
    loss = loss_fn(logits, labels_train)  # compute the loss
    losses.append(loss)
    if (i + 1) % 100 == 0:
        print(f'epoch: {i + 1:2}  loss: {loss.item():10.8f}')
    optimizer.zero_grad()  # clear the gradients
    loss.backward()  # compute the gradients via backpropagation
    optimizer.step()  # update the weights using the gradients

# save the model
torch.save(model.state_dict(), './model.pt')

# test the model on the unseen test data
predictions = []
text_predictions = []
probabilities = []
with torch.no_grad():  # don't need gradients here as we are no longer training
    for feature in features_test:
        logits = model(feature)  # pass the features forward through the model
        probs = F.softmax(logits, dim=-1)  # convert the logits into probabilities
        prediction = predict_by_max_logit(logits)  # make prediction on the class that has the highest value
        predictions.append(prediction)  # make prediction on the class that has the highest value
        text_predictions.append(coarse_label_map[prediction.numpy()])
        probabilities.append(probs)

predictions = torch.hstack(predictions)
probabilities = torch.vstack(probabilities)
if labels_test[0] > -1:  # we have test labels
    plot_confusion_matrix(predictions.numpy(), labels_test.numpy(), coarse_label_map)
    list_errors(predictions.numpy(), labels_test.numpy(), aors_test, aor_indices_test, probabilities.numpy(), coarse_label_map)
    print("Accuracy = {0:0.1f}%".format(accuracy_score(labels_test.numpy(), predictions.numpy()) * 100.0))
else:  # we don't have test labels
    for prediction, aor, index in zip(text_predictions, aors_test, aor_indices_test):
        print("AOR: {}, Pointing: {}, Predicted Class: {}, Probability: {}".format(aor, index, prediction, max(probs.numpy())))
    df = pd.DataFrame({'aor': aors_test, 'pointing': aor_indices_test, 'predicted class': text_predictions})
    df.to_csv( "predictions.csv", header=False, index=False)