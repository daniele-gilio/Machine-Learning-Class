import numpy as np
import cupy as cp
import matplotlib.pyplot as plt


def multinomial_logreg_inference(X, W, b, convert=True):
    """Predict class probabilities.
    Parameters
    ----------
    X : ndarray, shape (m, n)
         icput features (one row per feature vector).
    W : ndarray, shape (n, k)
         weight vectors, each row representing a different class.
    b : ndarray, shape (k,)
         vector of biases.
    Returns
    -------
    P : ndarray, shape (m, k)
         probability estimates.
    """
    X=cp.array(X)
    W=cp.array(W)
    b=cp.array(b)
    logits = X @ W + b.T
    if convert==True:
        return cp.asnumpy(softmax(logits))
    else:
        return softmax(logits)


def softmax(Z):
    """Softmax operator.
    Parameters
    ----------
    Z : ndarray, shape (m, n)
         icput vectors.
    Returns
    -------
    ndarray, shape (m, n)
         data after the softmax has been applied to each row.
    """
    # Subtracting the maximum improves numerical stability
    E = cp.exp(Z - Z.max(1, keepdims=True))
    return E / E.sum(1, keepdims=True)


def one_hot_vectors(Y, classes):
    """Convert an array of labels into a matrix of one-hot vectors.
    Parameters
    ----------
    Y : ndarray, shape (m,)
         labels.
    classes : int
         number of classes.  If None it is deduced from Y.
    Returns
    -------
    ndarray, shape (m, classes)
         One-hot vectors representing the labels Y.
    """
    m = Y.shape[0]
    H = cp.zeros((m, classes))
    H[cp.arange(m), Y] = 1
    return H


def multinomial_logreg_train(X, Y, lambda_, lr=1e-3, steps=1000,
                             init_w=None, init_b=None, x_val=None, y_val=None):
    """Train a classifier based on multinomial logistic regression.
    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        training labels with integer values in the range 0...(k-1).
    lambda_ : float
        regularization coefficient.
    lr : float
        learning rate
    steps : int
        number of training steps
    init_w : ndarray, shape (n, k)
        initial weights (None for zero initialization)
    init_b : ndarray, shape (k,)
        initial biases (None for zero initialization)
    Returns
    -------
    w : ndarray, shape (n, k)
        learned weights (one vector per class).
    b : ndarray, shape (k,)
        vector of biases.
    """
    m, n = X.shape
    m=int(m)
    n=int(n)
    k = Y.max() + 1
    k=int(k)
    X=cp.array(X)
    Y=cp.array(Y)
    W = (cp.array(init_w) if init_w is not None else cp.zeros((n, k)))
    b = (cp.array(init_b) if init_b is not None else cp.zeros(k))
    H = one_hot_vectors(Y, k)
    train_accs=[]
    val_accs=[]
    ep=[]
    if x_val is not None and y_val is not None:
        x_val=cp.array(x_val)
        y_val=cp.array(y_val)
        plt.ion()
        for step in range(steps):
            P = multinomial_logreg_inference(X, W, b, convert=False)
            grad_W = (X.T @ (P - H)) / m + 2 * lambda_ * W
            grad_b = (P - H).mean(0)
            W -= lr * grad_W
            b -= lr * grad_b

            if (step+1)%1000==0:
                print("Step: ", step+1)
                P = multinomial_logreg_inference(X, W, b, convert=False)
                labels=cp.argmax(P, axis=1)
                acc=cp.array(labels==Y).mean()*100
                train_accs.append(acc)
                scores=multinomial_logreg_inference(x_val, W, b, convert=False)
                labels=cp.argmax(scores, axis=1)
                acc=cp.array(labels==y_val).mean()*100
                val_accs.append(acc)
                ep.append(step)
                plt.clf()
                plt.plot(ep, train_accs, label="Training")
                plt.plot(ep, val_accs, label="Validation")
                plt.grid(1)
                plt.legend()
                plt.pause(0.005)
                plt.clf()

        plt.ioff()
        plt.plot(ep, train_accs, label="Training")
        plt.plot(ep, val_accs, label="Validation")
        plt.grid(1)
        plt.legend()
        plt.title("LogReg Training")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy [%]")
        plt.savefig("logreg_training.png")
        plt.clf()

    else:
        for step in range(steps):
            P = multinomial_logreg_inference(X, W, b, convert=False)
            grad_W = (X.T @ (P - H)) / m + 2 * lambda_ * W
            grad_b = (P - H).mean(0)
            W -= lr * grad_W
            b -= lr * grad_b

            if (step+1)%1000==0:
                print("Step: ", step+1)
    return cp.asnumpy(W), cp.asnumpy(b)


def cross_entropy(H, P):
    """Average cross entropy.
    Parameters
    ----------
    H : ndarray, shape (m, k)
        one hot vectors for the target labels.
    P : ndarray, shape (m, k)
        probability estimates.
    Returns
    -------
    float
        average cross entropy.
    """
    return -(H * cp.nan_to_num(cp.log(P))).sum(1).mean()
