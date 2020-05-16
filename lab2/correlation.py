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

def test_norm(times):
    train_data=np.loadtxt(data_path+"train"+ext)
    x_train=np.array(train_data[:,:-1])
    y_train=np.array(train_data[:,-1].astype(np.int32))

    val_data=np.loadtxt(data_path+"validation"+ext)
    x_val=np.array(val_data[:,:-1])
    y_val=np.array(val_data[:,-1].astype(np.int32))

    mvar_train, mvar_val = meanvar_normalization(x_train, x_val)
    minmax_train, minmax_val = minmax_normalization(x_train, x_val)
    maxabs_train, maxabs_val = maxabs_normalization(x_train, x_val)
    l1_train, l1_val = l1_normalization(x_train, x_val)
    l2_train, l2_val = l2_normalization(x_train, x_val)
    white_train, white_val = whitening(x_train, x_val)

    train_norm=np.array([x_train, mvar_train, minmax_train, maxabs_train, l1_train, l2_train, white_train])
    val_norm=np.array([x_val, mvar_val, minmax_val, maxabs_val, l1_val, l2_val, white_val])
    names=["No-Norm", "MeanVar", "MinMax", "MaxAbs", "l1", "l2", "Whitening"]
    indexes=np.random.randint(0,1076,120)
    train_sample=np.zeros((120,1024))
    r=np.zeros(7)
    for l in range(times):
        for k in range(7):
            for i in range(120):
                train_sample[i]=train_norm[k][indexes[i]]
            r[k]+=np.cov(train_sample, val_norm[k]).mean()/(train_sample.std()*val_norm[k].std())
            #print(names[k]+":", r[k])
            #print("Done with", names[k], k+1, "/ 6")
        if (l+1)%100==0:
            print("Done with test", l+1)
    for l in range(7):
        print(names[l], r[l]/times)

data_path="spoken-digits/"
ext=".txt.gz"
test=False #True if one wants to test all the available normalization techniques
image_save=True
visual=False

test_norm(1000)
