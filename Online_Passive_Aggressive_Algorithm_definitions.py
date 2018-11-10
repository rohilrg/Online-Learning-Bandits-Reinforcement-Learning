__author__ = 'rohil'

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def passive_agrresive_algorithm(train_x,train_y,type_of_update=None,C = 0.01):

    no_of_features = train_x.shape[1]

    # Initialize the weights vectors with shape equal to number of features
    weights = np.zeros((no_of_features, 1))

    ## Passive Aggressive Algorithm

    prediction_error = []
    for idx, rows in train_x.iterrows():

        rows = np.array(rows).reshape((1,no_of_features))
        y_prediction = np.sign(np.dot(weights.T, rows.T))
        c = train_y[idx]- y_prediction
        prediction_error.append(c)
        fg = (train_y[idx] * np.dot(weights.T, rows.T))
        loss = max(0, 1 - (train_y[idx] * np.dot(weights.T, rows.T)))

        norm_square = np.power(np.linalg.norm(rows, ord=2), 2)

        tau = 0
        if type_of_update == 'cu':
            tau = loss/ norm_square

        if type_of_update == 'fr':
            tau = min(C,loss/norm_square)

        if type_of_update == 'sr':
            tau = loss/(norm_square+(1/2*C))

        coeff = tau*train_y[idx]

        weights+=coeff*rows.T

    return prediction_error, weights

if __name__ == "__main__":
    # np.random.seed(1000)
    #
    # nb_samples = 500
    # nb_features = 4
    #
    # # Create the dataset
    # X, Y = make_classification(n_samples=nb_samples,
    #                            n_features=nb_features,
    #                            n_informative=nb_features - 2,
    #                            n_redundant=0,
    #                            n_repeated=0,
    #                            n_classes=2,
    #                            n_clusters_per_class=2)
    #
    # # Split the dataset
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=1000)
    #
    # Y_train[Y_train == 0] = -1
    # X_train=pd.DataFrame(X_train)
    # predicition_error = passive_agrresive_algorithm(X_train, Y_train, type_of_update='sr',C=100)
    #
    # print(predicition_error)

    df = pd.read_csv('diabetes_scale.csv')
    df = df.set_index('Sno', drop=True)
    train_X = df.iloc[:, 0:8]
    train_Y = df.iloc[:, 8:9]
    train_Y = np.ravel(train_Y)
    pred_err, weights = passive_agrresive_algorithm(train_X, train_Y, type_of_update='cu', C=10)

    c = np.count_nonzero(pred_err, axis=0)
    print(c)
    print('PA accuracy: {}'.format(1 - float(c) / train_X.shape[0]))
