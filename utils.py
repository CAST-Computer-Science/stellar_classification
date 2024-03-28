import torch
import random
import numpy as np


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def predict_by_max_logit(logits):
    return torch.argmax(logits, dim=-1)


def list_errors(predictions, labels, aors, aor_indices, probs, label_map):
    for prediction, label, aor, aor_index, prob in zip(predictions, labels, aors, aor_indices, probs):
        if prediction != label:
            print('-----------------------------------------------------------------')
            if aor_index == 'None':
                print('AOR {} {} was misclassified as {}.'.format(aor, label_map[label], label_map[prediction]))
            else:
                print('AOR {} index {} {} was misclassified as {}.'.format(aor, aor_index, label_map[label], label_map[prediction]))
            print('Probabilities:')
            for i, item in enumerate(label_map):
                print('{}: {:1.5f}'.format(item, prob[i]))


def normalise_features(features):
    length = len(features)
    min = np.min(features)
    max = np.max(features)
    diff = max - min

    for i in range(length):
        features[i] = np.divide(features[i] - min, diff)
    return features
