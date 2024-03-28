import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def make_loss_plot(epochs, losses):
    plt.plot(range(epochs), torch.hstack(losses).detach().numpy())
    plt.ylabel('Loss')
    plt.xlabel('Epoch');
    plt.savefig('plot_losses.pdf')
    plt.show()


def plot_features(features, labels, label_map):
    tsne = TSNE(n_components=2, init='pca').fit_transform(features)
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    fig = plt.figure(figsize=(14, 8), layout="constrained")
    ax = fig.add_subplot(111)

    colors_per_class = [
        'blue', 'orange', 'green', 'red', 'yellow', 'purple', 'brown',
        'pink', 'gray', 'olive', 'cyan', 'black', 'lime', 'linen', 'royalblue',
        'magenta', 'gold', 'tan', 'lightskyblue', 'salmon', 'slategray', 'teal'
    ]

    # for every class, we'll add a scatter plot separately
    for label in np.arange(0, len(label_map)):
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format
        color = colors_per_class[label]

        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label_map[label])

    # build a legend using the labels we set previously
    fig.legend(loc='outside upper center', ncols=6)
    plt.show()


def plot_confusion_matrix(predictions, labels, label_map):
    cm = confusion_matrix(labels, predictions, labels=np.arange(0, len(label_map)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_map)
    disp.plot(xticks_rotation=90)
    plt.show()