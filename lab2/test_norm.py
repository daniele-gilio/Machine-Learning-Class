#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

######## Normalization Functions ##########

def meanvar_normalization(X, *Xtest):
    """Normalize features using moments.
    Linearly normalize each input feature to make it have zero mean
    and unit variance.  Test features, when given, are scaled using
    the statistics computed on X.
    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    Xtest : ndarray, shape (mtest, n) or None
         zero or more arrays of test features (one row per feature vector).
    Returns
    -------
    ndarray, shape (m, n)
        normalized features.
    ndarray, shape (mtest, n)
        normalized test features (one for each array in Xtest).
    """
    mu = X.mean(0, keepdims=True)
    sigma = X.std(0, keepdims=True)
    X = X - mu
    X /= np.maximum(sigma, 1e-15)  # 1e-15 avoids division by zero
    if not Xtest:
        return X
    Xtest = tuple((Xt - mu) / np.maximum(sigma, 1e-15) for Xt in Xtest)
    return (X,) + Xtest


def minmax_normalization(X, *Xtest):
    """Scale features in the [0, 1] range.
    Linearly normalize each input feature in the [0, 1] range.  Test
    features, when given, are scaled using the statistics computed on
    X.
    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    Xtest : ndarray, shape (mtest, n) or None
         zero or more arrays of test features (one row per feature vector).
    Returns
    -------
    ndarray, shape (m, n)
        normalized features.
    ndarray, shape (mtest, n)
        normalized test features (one for each array in Xtest).
    """
    xmin = X.min(0, keepdims=True)
    xmax = X.max(0, keepdims=True)
    X = X - xmin
    X /= np.maximum(xmax - xmin, 1e-15)  # 1e-15 avoids division by zero
    if not Xtest:
        return X
    Xtest = tuple((Xt - xmin) / np.maximum(xmax - xmin, 1e-15) for Xt in Xtest)
    return (X,) + Xtest


def maxabs_normalization(X, *Xtest):
    """Scale features in the [-1, 1] range.
    Linearly normalize each input feature in the [-1, 1] range by
    dividing them by the maximum absolute value.  Test features, when
    given, are scaled using the statistics computed on X.
    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    Xtest : ndarray, shape (mtest, n) or None
         zero or more arrays of test features (one row per feature vector).
    Returns
    -------
    ndarray, shape (m, n)
        normalized features.
    ndarray, shape (mtest, n)
        normalized test features (one for each array in Xtest).
    """
    # 1e-15 avoids division by zero
    amax = np.maximum(np.abs(X).max(0, keepdims=True), 1e-15)
    X = X / amax
    if not Xtest:
        return X
    Xtest = tuple(Xt / amax for Xt in Xtest)
    return (X,) + Xtest


def l2_normalization(X, *Xtest):
    """L2 normalization of feature vectors.
    Scale feature vectors to make it have unit Euclidean norm.  Test
    features, when given, are scaled as well.
    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    Xtest : ndarray, shape (mtest, n) or None
         zero or more arrays of test features (one row per feature vector).
    Returns
    -------
    ndarray, shape (m, n)
        normalized features.
    ndarray, shape (mtest, n)
        normalized test features (one for each array in Xtest).
    """
    q = np.sqrt((X ** 2).sum(1, keepdims=True))
    X = X / np.maximum(q, 1e-15)  # 1e-15 avoids division by zero
    if not Xtest:
        return X
    Xtest = tuple(l2_normalization(Xt) for Xt in Xtest)
    return (X,) + Xtest


def l1_normalization(X, *Xtest):
    """L1 normalization of feature vectors.
    Scale feature vectors to make it have unit L1 norm.  Test
    features, when given, are scaled as well.
    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    Xtest : ndarray, shape (mtest, n) or None
         zero or more arrays of test features (one row per feature vector).
    Returns
    -------
    ndarray, shape (m, n)
        normalized features.
    ndarray, shape (mtest, n)
        normalized test features (one for each array in Xtest).
    """
    q = np.abs(X).sum(1, keepdims=True)
    X = X / np.maximum(q, 1e-15)  # 1e-15 avoids division by zero
    if not Xtest:
        return X
    Xtest = tuple(l1_normalization(Xt) for Xt in Xtest)
    return (X,) + Xtest


def whitening(X, *Xtest):
    """Whitening transform.
    Linearly transform features to make it have zero mean, unit
    variance and null covariance.  Test features, when given, are
    trandformed using the statistics computed on X.
    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    Xtest : ndarray, shape (mtest, n) or None
         zero or more arrays of test features (one row per feature vector).
    Returns
    -------
    ndarray, shape (m, n)
        normalized features.
    ndarray, shape (mtest, n)
        normalized test features (one for each array in Xtest).
    """
    mu = X.mean(0, keepdims=True)
    sigma = np.cov(X.T)
    evals, evecs = np.linalg.eig((sigma))
    evals=np.array(evals)
    evecs=np.array(evecs)
    w = (np.maximum(evals, 1e-15) ** -0.5) * evecs  # 1e-15 avoids div. by zero
    X = (X - mu) @ w
    if not Xtest:
        return X
    Xtest = tuple((Xt - mu) @ w for Xt in Xtest)
    return (X,) + Xtest


########### Multi-Layer Perceptron Functions and Classes ###########

def softmax(Z):
    """Softmax operator.
    Parameters
    ----------
    Z : ndarray, shape (m, n)
         input vectors.
    Returns
    -------
    ndarray, shape (m, n)
         data after the softmax has been applied to each row.
    """
    # Subtracting the maximum improves numerical stability
    E = np.exp(Z - Z.max(1, keepdims=True))
    return E / E.sum(1, keepdims=True)


class MLP:
    """Multi-layer perceptron.
    A multi-layer perceptron for classification.
    The activation function for the output layer is the softmax
    operator, while hidden neurons use relu.  The loss function used
    during training is the cross entropy.
    To use a different architecture it is possible to define a new
    derived class which overrides some of the methods.
    """

    def __init__(self, neuron_counts):
        """Create and initialiaze the MLP.
        At least two layers must be specified (input and output).
        Parameters
        ----------
        neuron_counts : list
            number of neurons in the layers (first input, then hiddens,
            then output).
        """
        # Initialize weights with the Kaiming technique.
        self.weights = [np.random.randn(m, n) * np.sqrt(2.0 / m)
                        for m, n in zip(neuron_counts[:-1], neuron_counts[1:])]
        # Biases are zero-initialized.
        self.biases = [np.zeros(m) for m in neuron_counts[1:]]
        # Accumulators for the momentum terms
        self.update_w = [np.zeros_like(w) for w in self.weights]
        self.update_b = [np.zeros_like(b) for b in self.biases]
        self.global_counter=0

    def forward(self, X):
        """Compute the activations of the neurons.
        Parameters
        ----------
        X : ndarray, shape (m, n)
            input features (one row per feature vector).
        Returns
        -------
        list
            the list of activations of the neurons
        The returned list contains arrays (one per layer) with the
        activations of the layers.  The first element of the list is
        the input X, followed by the activations of hidden layers and
        trminated by the activations in the output layer.
        """
        activations = [X]
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            X = self.forward_hidden_layer(X, W, b)
            X = self.forward_hidden_activation(X)
            activations.append(X)
        X = self.forward_output_layer(X, self.weights[-1], self.biases[-1])
        X = self.forward_output_activation(X)
        activations.append(X)
        return activations

    def backward(self, Y, activations):
        """Compute the derivatives of the loss wrt the activations.
        Parameters
        ----------
        Y : ndarray, shape (m,)
            target output (integer class labels).
        activations : list
            activations computed by the forward method.
        Returns
        -------
        list
            the list of computed derivatives, one for each layer.
        """
        d = self.backward_output_activation(Y, activations[-1])
        derivatives = [d]
        if len(self.weights) > 1:
            d = self.backward_output_layer(self.weights[-1],
                                           self.biases[-1], d)
            d = self.backward_hidden_activation(activations[-2], d)
            derivatives.append(d)
        for W, b, X in zip(self.weights[-2:0:-1], self.biases[-2:0:-1],
                           activations[-3::-1]):
            d = self.backward_hidden_layer(W, b, d)
            d = self.backward_hidden_activation(X, d)
            derivatives.append(d)
        return derivatives[::-1]

    def backpropagation(self, X, Y, lr=1e-4, lambda_=1e-5, momentum=0.99):
        """Backpropagation algorithm.
        Perform both the forward and the backward steps and update the
        parameters.
        Parameters
        ----------
        X : ndarray, shape (m, n)
            input features (one row per feature vector).
        Y : ndarray, shape (m,)
            target output (integer class labels).
        lr : float
            learning rate.
        lambda_ : float
            regularization coefficients.
        momentum : float
            momentum coefficient.
        """
        activations = self.forward(X)
        derivatives = self.backward(Y, activations)
        for X, D, W, b, uw, ub in zip(activations, derivatives,
                                      self.weights, self.biases,
                                      self.update_w, self.update_b):
            grad_W = (X.T @ D) + lambda_ * W
            grad_b = D.sum(0)
            uw *= momentum
            uw -= lr * grad_W
            W += uw
            ub *= momentum
            ub -= lr * grad_b
            b += ub

    def inference(self, X):
        """Compute the predictions of the network.
        Parameters
        ----------
        X : ndarray, shape (m, n)
            input features (one row per feature vector).
        Returns
        -------
        ndarray, shape (m,)
            predicted labels, in the range 0, ..., k - 1
        ndarray, shape (m, k)
            posterior probability estimates.
        """
        probs = self.forward(X)[-1]
        labels = np.argmax(probs, 1)
        return labels, probs

    def train(self, X, Y, lr0=1e-4, lambda_=1e-5, momentum=0.99,
              steps=10000, batch=None):
        """Train the network.
        Apply multiple steps of stochastic gradient descent.
        Parameters
        ----------
        X : ndarray, shape (m, n)
            input features (one row per feature vector).
        Y : ndarray, shape (m,)
            target output (integer class labels).
        lr : float
            learning rate.
        lambda_ : float
            regularization coefficients.
        momentum : float
            momentum coefficient.
        steps : int
            training iterations.
        batch : int or None
            size of the minibatch used in each step.  When None all
            the data is used in each step.
        """
        m = X.shape[0]
        if batch is None:
            batch = X.shape[0]
        i = m
        indices = np.arange(m)
        for step in range(steps):
            if self.global_counter%steps==0:
                lr=lr0/np.sqrt(self.global_counter/steps + 1) #update at every epoch
            if i + batch > m:
                i = 0
                np.random.shuffle(indices)
            self.backpropagation(X[indices[i:i + batch], :],
                                 Y[indices[i:i + batch]],
                                 lr=lr,
                                 lambda_=lambda_,
                                 momentum=momentum)
            i += batch
            self.global_counter+=1

    def save(self, filename):
        """Save the network to the file."""
        np.savez(filename, weights=self.weights, biases=self.biases)

    @classmethod
    def load(cls, filename):
        """Create a new network from the data saved in the file."""
        data = np.load(filename, allow_pickle=True)
        neurons = [w.shape[0] for w in data["weights"]]
        neurons.append(data["weights"][-1].shape[1])
        network = cls(neurons)
        network.weights = data["weights"]
        network.biases = data["biases"]
        return network

    # These last methods can be modified by derived classes to change
    # the architecture of the MLP.

    def forward_hidden_layer(self, X, W, b):
        """Forward pass of hidden layers."""
        return X @ W + b

    def forward_hidden_activation(self, X):
        """Activation function of hidden layers."""
        return relu(X)

    def forward_output_layer(self, X, W, b):
        """Forward pass of the output layer."""
        return X @ W + b

    def forward_output_activation(self, X):
        """Activation function of the output layer."""
        return softmax(X)

    def backward_hidden_layer(self, W, b, d):
        """Backward pass of hidden layers."""
        return d @ W.T

    def backward_hidden_activation(self, Y, d):
        """Derivative of the activation function of hidden layers."""
        return d * (Y > 0).astype(int)

    def backward_output_layer(self, W, b, d):
        """Backward pass of the ouput layer."""
        return d @ W.T

    def backward_output_activation(self, Y, P):
        """Derivative of the activation function of output layer."""
        d = P.copy()
        # Implicitly subtract the one-hot vectors
        d[np.arange(Y.shape[0]), Y] -= 1
        return d / Y.shape[0]

    def loss(self, Y, P):
        """Compute the average cross-entropy."""
        return -np.log(P[np.arange(Y.shape[0]), Y]).mean()


def relu(x):
    """ReLU activation function."""
    return np.maximum(x, 0)


def test_norm(data_path, epochs, image_save=True):
    ext=".txt.gz"
    train_data=np.loadtxt(data_path+"train"+ext)
    x_train=np.array(train_data[:,:-1])
    y_train=np.array(train_data[:,-1].astype(np.int32))

    plt.imshow((x_train[0]).reshape(16,64), cmap="gist_heat")
    plt.title("Digit 5 Spectrogram without normalization")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    if image_save==True:
        plt.savefig("nonorm_data.png")
        plt.clf()
    else:
        plt.show()

    val_data=np.loadtxt(data_path+"validation"+ext)
    x_val=np.array(val_data[:,:-1])
    y_val=np.array(val_data[:,-1].astype(np.int32))

    mvar_train, mvar_val = meanvar_normalization(x_train, x_val)
    minmax_train, minmax_val = minmax_normalization(x_train, x_val)
    maxabs_train, maxabs_val = maxabs_normalization(x_train, x_val)
    l1_train, l1_val = l1_normalization(x_train, x_val)
    l2_train, l2_val = l2_normalization(x_train, x_val)
    white_train, white_val = whitening(x_train, x_val)

    train_norm=np.array([mvar_train, minmax_train, maxabs_train, l1_train, l2_train, white_train])
    val_norm=np.array([mvar_val, minmax_val, maxabs_val, l1_val, l2_val, white_val])
    names=["MeanVar", "MinMax", "MaxAbs", "l1", "l2", "Whitening"]
    for k in range(6):
        nn_multi=MLP([1024, 256, 128, 32, 10])
        batch_size=256
        steps=len(x_train)//batch_size
        train_accs=[]
        val_accs=[]
        ep_vec=[]
        for i in range(epochs):
            nn_multi.train(train_norm[k], y_train, lr0=1e-2, lambda_=1e-5, momentum=0.99,
                          steps=steps, batch=batch_size)
            train_labels=nn_multi.inference(train_norm[k])[0]
            val_labels=nn_multi.inference(val_norm[k])[0]
            train_acc=(train_labels==y_train).mean()*100
            val_acc=(val_labels==y_val).mean()*100
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            ep_vec.append(i)
        np.savetxt(names[k]+ ".txt", np.column_stack((ep_vec, train_accs, val_accs)))
        plt.imshow(((train_norm[k])[0]).reshape(16,64), cmap="gist_heat")
        plt.title("Digit 5 Spectrogram with "+ names[k]+" normalization")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        if image_save==True:
            plt.savefig(names[k]+"_data.png")
            plt.clf()
        else:
            plt.show()

        print("Done with", names[k], k+1, "/ 6")

    ###### Create Plots Images ######

    for name in names:
        x, train = np.loadtxt(name+".txt", unpack=True, usecols=(0,1))
        plt.plot(x, train, label=name)
    plt.legend()
    plt.grid()
    plt.title("Training Accuracies")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy [%]")
    if image_save==True:
        plt.savefig("training_accs.png")
    else:
        plt.show()

    plt.clf()

    for name in names:
        x, val = np.loadtxt(name+".txt", unpack=True, usecols=(0,2))
        plt.plot(x, val, label=name)
    plt.legend()
    plt.grid()
    plt.title("Validation Accuracies")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy [%]")
    if image_save==True:
        plt.savefig("val_accs.png")
    else:
        plt.show()

    plt.clf()

data_path="spoken-digits/"
epochs=1000
test_norm(data_path, epochs)
