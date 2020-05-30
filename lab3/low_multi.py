from pvml import multinomial_logistic, mlp
import image_features
import glob
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

def file_list(path):
    files = [f for f in glob.glob(path + "/" + "**/*.jpg", recursive=True)]
    return files

load=True
aug=True
if aug==True:
    print("Using Augmented Dataset")
classes=["bluebell", "buttercup", "colts-foot", "daisy", "dandelion", "fritillary",
            "iris", "lily-valley", "pansy", "sunflower", "tigerlily", "windflower"]

test_path="flowers/test"
n_c=len(classes)
test_file=file_list(test_path)
test_size=20*12

print("Loading Test Data...")
im=image.imread(test_file[0])
x_test=image_features.color_histogram(im).flatten()
x_test=np.concatenate((x_test, image_features.edge_direction_histogram(im).flatten()))
x_test=np.concatenate((x_test, image_features.cooccurrence_matrix(im).flatten()))
x_test=np.concatenate((x_test, image_features.rgb_cooccurrence_matrix(im).flatten()))
y_test=np.zeros(test_size)

for c in classes:
    if c in test_file[0]:
        y_test[0]=classes.index(c)

counter=1

for f in test_file[1:]:
    im=image.imread(f)
    vec=image_features.color_histogram(im).flatten()
    vec=np.concatenate((vec, image_features.edge_direction_histogram(im).flatten()))
    vec=np.concatenate((vec, image_features.cooccurrence_matrix(im).flatten()))
    vec=np.concatenate((vec, image_features.rgb_cooccurrence_matrix(im).flatten()))
    vec=vec[None, :]
    x_test=np.append(x_test, vec)
    for c in classes:
        if c in f:
            y_test[counter]=classes.index(c)
    counter+=1

x_test=x_test.reshape((test_size,len(x_test)//test_size))
y_test=y_test.astype(np.int32)

print("Done")

if load==False:
    train_path="flowers/train"
    a_train_path="flowers/train_augmented/"
    if aug==False:
        train_file=file_list(train_path)
        train_size=60*12
    else:
        train_file=file_list(a_train_path)
        train_size=60*12*18

    print("Loading Training Data...")
    im=image.imread(train_file[0])
    x_train=image_features.color_histogram(im).flatten()
    x_train=np.concatenate((x_train, image_features.edge_direction_histogram(im).flatten()))
    x_train=np.concatenate((x_train, image_features.cooccurrence_matrix(im).flatten()))
    x_train=np.concatenate((x_train, image_features.rgb_cooccurrence_matrix(im).flatten()))

    x_train=x_train[None,:]

    y_train=np.zeros(train_size)
    for c in classes:
        if c in train_file[0]:
            y_train[0]=classes.index(c)

    counter=1

    for f in train_file[1:]:
        im=image.imread(f)
        vec=image_features.color_histogram(im).flatten()
        vec=np.concatenate((vec, image_features.edge_direction_histogram(im).flatten()))
        vec=np.concatenate((vec, image_features.cooccurrence_matrix(im).flatten()))
        vec=np.concatenate((vec, image_features.rgb_cooccurrence_matrix(im).flatten()))
        vec=vec[None, :]
        x_train=np.append(x_train, vec)
        for c in classes:
            if c in f:
                y_train[counter]=classes.index(c)
        counter+=1

    x_train=x_train.reshape((train_size, len(x_train)//train_size))

    y_train=y_train.astype(np.int32)

    print("Done")

    dnn=mlp.MLP([x_train.shape[1], 512, 128, 32, 12])
    epochs=500
    batch_size=16
    steps=len(x_train)//batch_size #Automatically adjust steps so that steps*batch_size is almost the number of samples
    train_accs=[]
    test_accs=[]
    ep_vec=[]
    plt.ion()
    for i in range(epochs):
            dnn.train(x_train, y_train, lr=0.001, lambda_=1e-5, momentum=0.99,
                          steps=steps, batch=batch_size)
            train_labels=dnn.inference(x_train)[0]
            train_acc=(train_labels==y_train).mean()*100
            train_accs.append(train_acc)
            test_labels=dnn.inference(x_test)[0]
            test_acc=(test_labels==y_test).mean()*100
            test_accs.append(test_acc)
            ep_vec.append(i)
            plt.clf()
            plt.plot(ep_vec, train_accs, label="Training Accuracy")
            plt.plot(ep_vec, test_accs, label="Test Accuracy")
            plt.pause(0.005)
    plt.ioff()
    if aug==False:
        dnn.save("low_mlp_multi_params.npz")
    else:
        dnn.save("low_mlp_multi_aug_params.npz")
    test_labels=dnn.inference(x_test)[0]
    acc=(test_labels==y_test).mean()*100
    print("MLP Test Accuracy: ", acc)

    nn=mlp.MLP([x_train.shape[1], 12])
    epochs=800
    batch_size=16
    steps=len(x_train)//batch_size #Automatically adjust steps so that steps*batch_size is almost the number of samples
    train_accs=[]
    test_accs=[]
    ep_vec=[]
    plt.ion()
    for i in range(epochs):
            nn.train(x_train, y_train, lr=0.001, lambda_=1e-5, momentum=0.99,
                          steps=steps, batch=batch_size)
            train_labels=nn.inference(x_train)[0]
            train_acc=(train_labels==y_train).mean()*100
            train_accs.append(train_acc)
            test_labels=nn.inference(x_test)[0]
            test_acc=(test_labels==y_test).mean()*100
            test_accs.append(test_acc)
            ep_vec.append(i)
            plt.clf()
            plt.plot(ep_vec, train_accs, label="Training Accuracy")
            plt.plot(ep_vec, test_accs, label="Test Accuracy")
            plt.pause(0.005)
    plt.ioff()
    if aug==False:
        nn.save("low_slp_multi_params.npz")
    else:
        nn.save("low_slp_multi_aug_params.npz")
    test_labels=nn.inference(x_test)[0]
    acc=(test_labels==y_test).mean()*100
    print("Single Layer Perceptron Test Accuracy: ", acc, "%")

else:
    if aug==False:
        dnn=mlp.MLP.load("low_mlp_multi_params.npz")
        nn=mlp.MLP.load("low_slp_multi_params.npz")
    else:
        dnn=mlp.MLP.load("low_mlp_multi_aug_params.npz")
        nn=mlp.MLP.load("low_slp_multi_aug_params.npz")

    dnn_test_labels, dnn_probs=dnn.inference(x_test)
    acc=(dnn_test_labels==y_test).mean()*100
    print("MLP Test Accuracy: ", acc)

    max=0
    ind=0
    for i in range(len(y_test)):
        if y_test[i]!=dnn_test_labels[i]:
            if dnn_probs[i, dnn_test_labels[i]] > max:
                max=dnn_probs[i, dnn_test_labels[i]]
                ind=i

    print("MLP Most Difficult Image to Classify: ", test_file[ind])
    print("Real Class: ", y_test[ind], "Predicted Class: ", dnn_test_labels[ind])
    print("Prediction Confidence: ", dnn_probs[ind, dnn_test_labels[ind]]*100, "%")

    nn_test_labels, nn_probs=nn.inference(x_test)
    acc=(nn_test_labels==y_test).mean()*100
    print()
    print("Single Layer Perceptron Test Accuracy: ", acc, "%")

    max=0
    ind=0
    for i in range(len(y_test)):
        if y_test[i]!=nn_test_labels[i]:
            if nn_probs[i, nn_test_labels[i]] > max:
                max=nn_probs[i, nn_test_labels[i]]
                ind=i

    print("SLP Most Difficult Image to Classify: ", test_file[ind])
    print("Real Class: ", y_test[ind], "Predicted Class: ", nn_test_labels[ind])
    print("Prediction Confidence: ", nn_probs[ind, nn_test_labels[ind]]*100, "%")

    dnn_conf_mat=np.zeros((12,12))
    nn_conf_mat=np.zeros((12,12))

    for i in range(len(y_test)):
        dnn_conf_mat[int(y_test[i])][int(dnn_test_labels[i])]+=1
        nn_conf_mat[int(y_test[i])][int(nn_test_labels[i])]+=1

    for i in range(12):
        dnn_conf_mat[i,:]/=dnn_conf_mat[i,:].sum()
        nn_conf_mat[i,:]/=nn_conf_mat[i,:].sum()

    sn.heatmap(dnn_conf_mat, annot=True, cmap="coolwarm")
    if aug==False:
        plt.title("MLP Confusion Matrix")
        plt.savefig("mlp_confusion_matrix.png")
    else:
        plt.title("MLP Confusion Matrix (Augmented Dataset)")
        plt.savefig("mlp_aug_confusion_matrix.png")

    plt.clf()

    sn.heatmap(nn_conf_mat, annot=True, cmap="coolwarm")
    if aug==False:
        plt.title("SLP Confusion Matrix")
        plt.savefig("slp_confusion_matrix.png")
    else:
        plt.title("SLP Confusion Matrix (Augmented Dataset)")
        plt.savefig("slp_aug_confusion_matrix.png")
