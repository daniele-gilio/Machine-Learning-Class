import numpy as np
import cupy as cp
import pandas as pd

def logreg_inference(X, w, b):
    """Predict class probabilities.
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
        probability estimates (one per feature vector).
    """
    logits = cp.add(cp.matmul(X, w), b)
    return 1 / (1 + cp.exp(-logits))


def binary_cross_entropy(Y, P):
    """Average cross entropy.
    Parameters
    ----------
    Y : ndarray, shape (m,)
        binary target labels (0 or 1).
    P : ndarray, shape (m,)
        probability estimates.
    Returns
    -------
    float
        average cross entropy.
    """
    eps = 1e-3
    P = cp.clip(P, eps, 1 - eps)  # This prevents overflows
    return -(Y * cp.log(P) + (1 - Y) * cp.log(1 - P)).mean()


def logreg_train(X, Y, lr=1e-3, steps=1000, init_w=None, init_b=0):
    """Train a binary classifier based on logistic regression.
    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        binary training labels.
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
    for step in range(steps):
        P = logreg_inference(X, w, b)
        grad_w = cp.multiply(cp.matmul((P - Y), X), 1/m)
        grad_b = (P - Y).mean()
        w -= cp.multiply(lr ,grad_w)
        b -= lr * grad_b
        if (step+1)%100==0:
            print("Step: ", step, ", Loss: ", binary_cross_entropy(Y,P))
    return w, b


def logreg_l2_train(X, Y, lambda_, lr=1e-3, steps=1000, init_w=None,
                    init_b=0, lr0=1):
    """Train a binary classifier based on L2-regularized logistic regression.
    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        binary training labels.
    lambda_ : float
        regularization coefficient.
    lr : float
        learning rate.
    steps : int
        number of training steps.
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
    for step in range(steps):
        lr=lr0/(step+1)**0.5
        P = logreg_inference(X, w, b)
        grad_w = cp.multiply(cp.matmul((P - Y), X), 1/m)+2*lambda_*w
        grad_b = (P - Y).mean()
        w -= cp.multiply(lr ,grad_w)
        b -= lr * grad_b
        if step%1000==0:
            print("Step: ", step, ", Loss: ", binary_cross_entropy(Y,P))
    return w, b


def logreg_l1_train(X, Y, lambda_, lr=1e-3, steps=1000, init_w=None, init_b=0, lr0=1):
    """Train a binary classifier based on L1-regularized logistic regression.
    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        binary training labels.
    lambda_ : float
        regularization coefficient.
    lr : float
        learning rate.
    steps : int
        number of training steps.
    loss : ndarray, shape (steps,)
        loss value after each training step.
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
    for step in range(steps):
        lr=lr0/(step+1)**0.5
        P = logreg_inference(X, w, b)
        grad_w = cp.multiply(cp.matmul((P - Y), X), 1/m)+lambda_*cp.sign(w)
        grad_b = (P - Y).mean()
        w -= cp.multiply(lr ,grad_w)
        b -= lr * grad_b
        if step%100==0:
            prob=logreg_inference(X,w,b)
            pred=cp.asarray(prob>0.5)
            accuracy=(pred==Y).mean()*100
            print("Step: ", step, ", Accuracy: ", accuracy)
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
w,b=logreg_l1_train(X,Y, 0.,steps=steps, lr0=1)
#print(b)
train_prob=(logreg_inference(X,w,b))

train_pred=cp.asarray(train_prob>0.5)
accuracy=(train_pred==Y).mean()*100
print("Training Accuracy: ", accuracy)

X=cp.array(val_data[:,:-1])
Y=cp.array(val_data[:,-1])
val_prob=logreg_inference(X,w,b)
val_pred=cp.asarray(val_prob>0.5)
accuracy=(val_pred==Y).mean()*100
print("Validation Accuracy: ", accuracy)

X=cp.array(test_data[:,:-1])
Y=cp.array(test_data[:,-1])
test_prob=logreg_inference(X,w,b)
test_pred=cp.asarray(test_prob>0.5)
accuracy=(test_pred==Y).mean()*100
print("Test Accuracy: ", accuracy)
