import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def euclidean_distance(X1, X2):
    return np.sqrt(np.sum(np.square(X1 - X2), axis=1))


def k_nearest_neighbors(X, X_train, k):
    return np.argsort(euclidean_distance(X, X_train))[:k]


def predict(X_test, X_train, y_train, k):
    y_preds = []
    for X in X_test:
        knn_indexes = k_nearest_neighbors(X, X_train, k)
        knn_classes = y_train[knn_indexes]
        counts = np.bincount(knn_classes)
        max_count = np.argmax(counts)
        y_preds.append(max_count)
    return np.array(y_preds)


def accuracy(y_preds, y_test):
    return np.sum(np.where(y_preds == y_test, 1, 0)) * 100 / len(y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', dest='dataset_path', action='store', type=str, help='path to dataset',
                        required=True)
    parser.add_argument('--k', dest='k', action='store', type=int, help='k value for kNN', default=3)
    parser.add_argument('--test_size', dest='test_size', action='store', type=float,
                        help='size of test data in fraction (Default: 0.2)', default=0.2)

    args = parser.parse_args()

    df = pd.read_csv(args.dataset_path)

    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)

    y_preds = predict(X_test, X_train, y_train, args.k)
    print('Accuracy: {:.2f}%'.format(accuracy(y_preds, y_test)))
