from pvml import multinomial_logistic, mlp
import image_features
import glob
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt

def file_list(path):
    files = [f for f in glob.glob(path + "/" + "**/*.jpg", recursive=True)]
    return files

load=False
aug=True

classes=["bluebell", "buttercup", "colts-foot", "daisy", "dandelion", "fritillary",
            "iris", "lily-valley", "pansy", "sunflower", "tigerlily", "windflower"]

test_path="flowers/test"
n_c=len(classes)

test_file=file_list(test_path)
rgb_co_shape=27*27
test_size=20*12

print("Loading Test Data...")

x_test=image_features.rgb_cooccurrence_matrix(image.imread(test_file[0]))
y_test=np.zeros(test_size)
for c in classes:
    if c in test_file[0]:
        y_test[0]=classes.index(c)

counter=1

for f in test_file[1:]:
    x_test=np.append(x_test, image_features.rgb_cooccurrence_matrix(image.imread(f)))
    for c in classes:
        if c in f:
            y_test[counter]=classes.index(c)
    counter+=1

x_test=x_test.reshape(test_size,rgb_co_shape)
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

    print("Loading Train Data...")
    x_train=image_features.rgb_cooccurrence_matrix(image.imread(train_file[0]))
    y_train=np.zeros(train_size)

    for c in classes:
        if c in train_file[0]:
            y_train[0]=classes.index(c)

    counter=1

    for f in train_file[1:]:
        x_train=np.append(x_train, image_features.rgb_cooccurrence_matrix(image.imread(f)))
        for c in classes:
            if c in f:
                y_train[counter]=classes.index(c)
        counter+=1

    x_train=x_train.reshape(train_size,rgb_co_shape)
    y_train=y_train.astype(np.int32)

    print("Done")

    dnn=mlp.MLP([rgb_co_shape, 512, 128, 32, 12])
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
        dnn.save("low_mlp_params.npz")
    else:
        dnn.save("low_mlp_aug_params.npz")
    test_labels=dnn.inference(x_test)[0]
    acc=(test_labels==y_test).mean()*100
    print("MLP Test Accuracy: ", acc)

    nn=mlp.MLP([rgb_co_shape, 12])
    epochs=500
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
        nn.save("low_slp_params.npz")
    else:
        nn.save("low_slp_aug_params.npz")
    test_labels=nn.inference(x_test)[0]
    acc=(test_labels==y_test).mean()*100
    print("Single Layer Perceptron Test Accuracy: ", acc)

else:
    if aug==False:
        dnn=mlp.MLP.load("low_mlp_params.npz")
        nn=mlp.MLP.load("low_slp_params.npz")
    else:
        dnn=mlp.MLP.load("low_mlp_aug_params.npz")
        nn=mlp.MLP.load("low_slp_aug_params.npz")
    test_labels=dnn.inference(x_test)[0]
    acc=(test_labels==y_test).mean()*100
    print("MLP Test Accuracy: ", acc)
    test_labels=nn.inference(x_test)[0]
    acc=(test_labels==y_test).mean()*100
    print("Single Layer Perceptron Test Accuracy: ", acc)
