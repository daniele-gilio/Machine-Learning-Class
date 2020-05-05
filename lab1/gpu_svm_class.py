import numpy as np
import cupy as cp
import pandas as pd

def svm_inference(X, w, b):
    """SVM prediction of the class labels.
    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    w : ndarray, shape (n,)
         weight vector.
    b : float
         scalar bias.
    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector).
    ndarray, shape (m,)
        classification scores (one per feature vector).
    """
    #logits = X @ w + b
    logits=cp.add(cp.matmul(X, w),b)
    labels = (logits > 0).astype(int)
    return labels, logits


def hinge_loss(labels, logits):
    """Average hinge loss.
    Parameters
    ----------
    labels : ndarray, shape (m,)
        binary target labels (0 or 1).
    logits : ndarray, shape (m,)
        classification scores (logits).
    Returns
    -------
    float
        average hinge loss.
    """
    loss = cp.maximum(0, 1 - (2 * labels - 1) * logits)
    return loss.mean()


def svm_train(X, Y, lambda_, lr=1e-3, steps=1000, init_w=None, init_b=0, lr0=1):
    """Train a binary SVM classifier.
    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        binary training labels.
    lambda_ : float
        regularization coefficient.
    lr : float
        learning rate
    steps : int
        number of training steps
    init_w : ndarray, shape (n,)
        initial weights (None for zero initialization)
    init_b : float
        initial bias
    Returns
    -------
    w : ndarray, shape (n,)
        learned weight vector.
    b : float
        learned bias.
    """
    m, n = X.shape
    w = (init_w if init_w is not None else cp.zeros(n))
    b = init_b
    C = 2*Y - 1
    for step in range(steps):
        lr=lr0/(step+1)**0.5
        labels, logits = svm_inference(X, w, b)
        hinge_diff = -C * ((C * logits) < 1)
        #grad_w = (hinge_diff @ X) / m + lambda_ * w
        grad_w = cp.matmul(hinge_diff, X)/m + lambda_*w
        grad_b = hinge_diff.mean()
        w -= lr*grad_w
        b -= lr*grad_b
        if (step+1)%100==0:
            print("Step: ", step+1, ", Accuracy: ", (labels==Y).mean()*100)
    return w, b

n, steps=np.loadtxt("parameters.txt", unpack=True, usecols=(0,3))
n=int(n)
steps=int(steps)
train_name= "bows/train_" + str(n)
test_name="bows/test_" + str(n)
val_name="bows/val_" + str(n)

print("Loading Training Data...")
train_data=pd.read_csv(train_name+".gz", compression="gzip", dtype=np.int32, sep=" ", header=None).to_numpy()
print("Loading Validation Data...")
val_data=pd.read_csv(val_name + ".gz", compression="gzip", dtype=np.int32, sep=" ", header=None).to_numpy()
print("Loading Test Data...")
test_data=pd.read_csv(test_name + ".gz", compression="gzip", dtype=np.int32, sep=" ", header=None).to_numpy()
print("Done loading data")

X=cp.array(train_data[:,:-1])
Y=cp.array(train_data[:,-1])

w,b=svm_train(X,Y, lambda_=0.0001, steps=steps, lr0=1.)
labels,scores=svm_inference(X,w,b)
accuracy=(labels==Y).mean()*100
print("Training Accuracy: ", accuracy)

X=cp.array(val_data[:,:-1])
Y=cp.array(val_data[:,-1])
labels,logits=svm_inference(X,w,b)
accuracy=(labels==Y).mean()*100
print("Validation Accuracy: ", accuracy)

X=cp.array(test_data[:,:-1])
Y=cp.array(test_data[:,-1])
labels,scores=svm_inference(X,w,b)
accuracy=(labels==Y).mean()*100
print("Test Accuracy: ", accuracy)
