import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def euclidean_distance(X1, X2):
    """
    Function to compute the euclidean distance between two vectors
    :param X1: Vector 1
    :param X2: Vector 2
    :return: Scalar euclidean distance between X1 and X2
    """
    return np.sqrt(np.sum(np.square(X1 - X2), axis=1))


def k_nearest_neighbors(X, X_train, k):
    """
    Function to return indexes of the k nearest neighbors of vector X given training data X_train based on
    euclidean distance
    :param X: Vector 1 whose neighbors are to be searched
    :param X_train: List of vectors as the training data
    :param k: number of nearest neighbors
    :return: k indexes of X_train which are closest to X
    """
    return np.argsort(euclidean_distance(X, X_train))[:k]


def predict(X_test, X_train, y_train, k):
    """
    Function to predict labels for X_test using kNN algorithm with X_train and y_train as training data
    and k number of nearest neighbors
    :param X_test: List of vectors to predict labels for
    :param X_train: List of vectors as training features
    :param y_train: Corresponding labels for X_train
    :param k: number of nearest neighbors
    :return: predicted labels for X_test
    """
    y_preds = []
    for X in X_test:  # For each test vector
        knn_indexes = k_nearest_neighbors(X, X_train, k)  # Get k indexes for k nearest neighbors
        knn_classes = y_train[knn_indexes]  # Get k corresponding classes of neighbors
        counts = np.bincount(knn_classes)  # Count the number of labels in labels
        max_count = np.argmax(counts)  # Get the maximum occurring label
        y_preds.append(max_count)  # Append the maximum occurring label as prediction
    return np.array(y_preds)


def accuracy(y_preds, y_test):
    """
    Function to calculate the accuracy of algorithm
    :param y_preds: predictions for test data
    :param y_test: actual labels for test data
    :return: accuracy in percentage
    """
    return np.sum(np.where(y_preds == y_test, 1, 0)) * 100 / len(y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', dest='dataset_path', type=str, help='path to dataset', required=True)
    parser.add_argument('--k', dest='k', type=int, help='k value for kNN (Default: 3)', default=3)
    parser.add_argument('--test_size', dest='test_size', type=float,
                        help='size of test data in fraction (Default: 0.2)', default=0.2)

    args = parser.parse_args()

    df = pd.read_csv(args.dataset_path)

    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    # Split the data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)

    # Predict labels for test data using knn
    y_preds = predict(X_test, X_train, y_train, args.k)
    # Print the accuracy
    print('Accuracy: {:.2f}%'.format(accuracy(y_preds, y_test)))
