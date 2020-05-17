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
        data = np.load(filename)
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


####### Data Visualization and Normalization Testing Functions ########


def visualize(data, n, data_path):
    data=data.reshape(len(data), 16, 64)
    f=open(data_path+"train-names.txt")
    names=f.read().split()
    f.close()
    #np.random.seed(0) #<- Use for testing purposes, it ensures repeatability
    for i in range(n):
        s=np.random.randint(0, len(data))
        plot_data=data[s]
        plt.imshow((plot_data), cmap="gist_heat")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.title(names[s])
        plt.show()

data_path="spoken-digits/"
ext=".txt.gz"
test=False #True if one wants to test all the available normalization techniques
image_save=True
visual=False
lr0=1e-2
batch_size=256

##### Load Data #####
train_data=np.loadtxt(data_path+"train"+ext)
x_train=np.array(train_data[:,:-1])
y_train=np.array(train_data[:,-1].astype(np.int32))

val_data=np.loadtxt(data_path+"validation"+ext)
x_val=np.array(val_data[:,:-1])
y_val=np.array(val_data[:,-1].astype(np.int32))

test_data=np.loadtxt(data_path+"test"+ext)
x_test=np.array(test_data[:,:-1])
y_test=np.array(test_data[:,-1].astype(np.int32))

if visual==True:
    visualize(x_train, 5, data_path)
print("Pre-Normalization Values")
print("Max: ",x_train.max())
print("Min: ", np.min(x_train[np.nonzero(x_train)]))
print("Mean: ", x_train.mean())
print("Standard Deviation: ", x_train.std())
print()

x_train, x_test, x_val=meanvar_normalization(x_train, x_test, x_val) #<- Normalize with meanvar
if visual==True:
    visualize(x_train,5,data_path)
print("Post MeanVar Normalization Values")
print("Max: ",x_train.max())
print("Min: ", np.min(x_train[np.nonzero(x_train)]))
print("Mean: ", x_train.mean())
print("Standard Deviation: ", x_train.std())
print()


##### Multi Layer Training and Evaluation #####
nn_multi=MLP([1024, 256, 128, 32, 10])
epochs=1000
steps=len(x_train)//batch_size #Automatically adjust steps so that steps*batch_size is almost the number of samples
train_accs=[]
val_accs=[]
ep_vec=[]
plt.ion()
for i in range(epochs):
    nn_multi.train(x_train, y_train, lr0=lr0, lambda_=1e-5, momentum=0.99,
                  steps=steps, batch=batch_size)
    train_labels=nn_multi.inference(x_train)[0]
    val_labels=nn_multi.inference(x_val)[0]
    train_acc=(train_labels==y_train).mean()*100
    val_acc=(val_labels==y_val).mean()*100
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    ep_vec.append(i)
    plt.clf()
    plt.plot(ep_vec, train_accs, label="Training Accuracy")
    plt.plot(ep_vec, val_accs, label="Validation Accuracy")
    plt.pause(0.005)

    if (i+1)%10==0:
        print("Train Accuracy: ", train_acc, "Validation Accuracy: ", val_acc)
plt.ioff()

labels_multi=nn_multi.inference(x_test)[0]
test_acc=(labels_multi==y_test).mean()*100
print("Test Accuracy: ", test_acc)
print()


######## Single Layer Perceptron Training ########

train_data=np.loadtxt(data_path+"train"+ext)
x_train=np.array(train_data[:,:-1])
y_train=np.array(train_data[:,-1].astype(np.int32))

val_data=np.loadtxt(data_path+"validation"+ext)
x_val=np.array(val_data[:,:-1])
y_val=np.array(val_data[:,-1].astype(np.int32))

test_data=np.loadtxt(data_path+"test"+ext)
x_test=np.array(test_data[:,:-1])
y_test=np.array(test_data[:,-1].astype(np.int32))

x_train, x_test, x_val=l2_normalization(x_train, x_test, x_val) #<- Normalize with L2

nn_single=MLP([1024, 10])
epochs=1000
batch_size=8
steps=len(x_train)//batch_size
train_accs=[]
val_accs=[]
ep_vec=[]
plt.ion()
for i in range(epochs):
    nn_single.train(x_train, y_train, lr0=lr0, lambda_=1e-5, momentum=0.99,
                  steps=steps, batch=batch_size)
    train_labels=nn_single.inference(x_train)[0]
    val_labels=nn_single.inference(x_val)[0]
    train_acc=(train_labels==y_train).mean()*100
    val_acc=(val_labels==y_val).mean()*100
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    ep_vec.append(i)
    plt.clf()
    plt.plot(ep_vec, train_accs, label="Training Accuracy")
    plt.plot(ep_vec, val_accs, label="Validation Accuracy")
    plt.pause(0.005)

    if (i+1)%100==0:
        print("Train Accuracy: ", train_acc, "Validation Accuracy: ", val_acc)
plt.ioff()

labels=nn_single.inference(x_test)[0]
test_acc=(labels==y_test).mean()*100
print("Test Accuracy: ", test_acc)
print()

########### Create, fill and visualize confusion matrices #########

confusion_matrix=np.zeros((10,10))
for i in range(len(labels)):
    confusion_matrix[int(y_test[i])][int(labels[i])]+=1

for i in range(10):
    if confusion_matrix[i, :].sum()!=0:
        confusion_matrix[i, :]/=confusion_matrix[i,:].sum() #<- Normalize each row of the confusion matrix

#plt.figure(figsize = (10,7))
sn.heatmap(confusion_matrix, annot=True, cmap="coolwarm")
plt.title("Single Layer Confusion Matrix")
if image_save==True:
    plt.savefig("single_confusion_matrix.png")
else:
    plt.show()

plt.clf()

confusion_matrix=np.zeros((10,10))
for i in range(len(labels)):
    confusion_matrix[int(y_test[i])][int(labels_multi[i])]+=1

for i in range(10):
    if confusion_matrix[i, :].sum()!=0:
        confusion_matrix[i, :]/=confusion_matrix[i,:].sum() #<- Normalize each row of the confusion matrix

#plt.figure(figsize = (10,7))
sn.heatmap(confusion_matrix, annot=True, cmap="coolwarm")
plt.title("Multi Layer Confusion Matrix")
if image_save==True:
    plt.savefig("multi_confusion_matrix.png")
else:
    plt.show()

plt.clf()

####### Single Layer Perceptron Weights Visualization ########

w=(nn_single.weights[0])
for i in range(10):
    plt.imshow(w[:,i].reshape(16,64), cmap="gist_heat")
    plt.title(str(i))
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    if image_save==True:
        plt.savefig(str(i)+".png")
        plt.clf()
    else:
        plt.show()
