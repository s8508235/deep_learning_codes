import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from lib import *
from sklearn.metrics import accuracy_score
import os, sys

module_path = os.path.abspath(os.path.join('.'))
sys.path.append(module_path)

class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        The seed of the pseudo random number generator.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.activation(X)
            print(y.shape)
            print("----------")
            print(output.shape)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        print (np.dot(X, self.w_[1:]).shape)
        print("**************")
        print (self.w_[0].shape)
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)


def main():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
            'machine-learning-databases/iris/iris.data', header=None)
    df.columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class label']
    display(df.head())

    X = df[['Petal length', 'Petal width']].values
    y = pd.factorize(df['Class label'])[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    print('#Training data points: {}'.format(X_train.shape[0]))
    print('#Testing data points: {}'.format(X_test.shape[0]))
    print('Class labels: {} (mapped from {}'.format(np.unique(y), np.unique(df['Class label'])))


    sc = StandardScaler()
    sc.fit(X_train)
    # no fit for X_test
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    ppn = Perceptron(n_iter=10, eta0=0.1, random_state=0)
    ppn.fit(X_train_std, y_train)

    y_pred = ppn.predict(X_test_std)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(X=X_combined_std, y=y_combined,
                        classifier=ppn, test_idx=range(len(y_train),
                                                        len(y_train) + len(y_test)))
    plt.xlabel('Petal length [Standardized]')
    plt.ylabel('Petal width [Standardized]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    if not os.path.exists('./output'):
        os.makedirs('./output')
    plt.savefig('./output/fig-perceptron-scikit.png', dpi=300)
    #plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    ada1 = AdalineGD(n_iter=20, eta=0.0001).fit(X_train_std, y_train)
    ax[0].plot(range(1, len(ada1.cost_) + 1), ada1.cost_, marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Sum-squared-error')
    ax[0].set_title('Adaline - Learning rate 0.0001')

    ada2 = AdalineGD(n_iter=20, eta=0.1).fit(X_train_std, y_train)
    ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(Sum-squared-error)')
    ax[1].set_title('Adaline - Learning rate 0.1')

    plt.tight_layout()
    plt.savefig('./output/fig-adaline-gd-overshoot.png', dpi=300)
    plt.show()

    ada = AdalineGD(n_iter=20, eta=0.01)
    ada.fit(X_train_std, y_train)

    # cost values

    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')

    plt.tight_layout()
    plt.savefig('./output/fig-adalin-gd-cost.png', dpi=300)
    plt.show()

    # testing accuracy

    y_pred = ada.predict(X_test_std)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

    # plot decision boundary 

    plot_decision_regions(X_combined_std, y_combined, 
                        classifier=ada, test_idx=range(len(y_train),
                                                        len(y_train) + len(y_test)))
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('Petal length [Standardized]')
    plt.ylabel('Petal width [Standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('./output/fig-adaline-gd-boundary.png', dpi=300)
    plt.show()

if __name__ =="__main__":
    main()