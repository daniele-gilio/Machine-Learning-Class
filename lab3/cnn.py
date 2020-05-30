from pvml import pvmlnet, cnn, mlp
import glob
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt

def file_list(path):
    files = [f for f in glob.glob(path + "/" + "**/*.jpg", recursive=True)]
    return files

aug=True
classes=["bluebell", "buttercup", "colts-foot", "daisy", "dandelion", "fritillary",
            "iris", "lily-valley", "pansy", "sunflower", "tigerlily", "windflower"]
train_path="flowers/train"
a_train_path="flowers/train_augmented/"
test_path="flowers/test"
n_c=len(classes)

if aug==False:
    train_file=file_list(train_path)
    train_size=60*12
else:
    train_file=file_list(a_train_path)
    train_size=60*12*18
test_file=file_list(test_path)
test_size=20*12

pvml_cnn=pvmlnet.PVMLNet.load("pvmlnet.npz")

x_train=pvml_cnn.forward(image.imread(train_file[0])[None, :, :, :]/255)[-3]
y_train=np.zeros(train_size)
for c in classes:
    if c in train_file[0]:
        y_train[0]=classes.index(c)

x_test=pvml_cnn.forward(image.imread(test_file[0])[None, :, :, :]/255)[-3]
y_test=np.zeros(test_size)
for c in classes:
    if c in test_file[0]:
        y_test[0]=classes.index(c)
counter=1

print("Loading Training Files...")
for f in train_file[1:]:
    x_train=np.append(x_train, pvml_cnn.forward(image.imread(f)[None, :, :, :]/255)[-3])
    for c in classes:
        if c in f:
            y_train[counter]=classes.index(c)
    counter+=1

#x_train=x_train.reshape(train_size,224,224,3)/255
x_train=x_train.reshape((train_size, len(x_train)//train_size))
y_train=y_train.astype(np.int32)

counter=1

print("Loading Test Files...")
for f in test_file[1:]:
    x_test=np.append(x_test, pvml_cnn.forward(image.imread(f)[None, :, :, :]/255)[-3])
    for c in classes:
        if c in f:
            y_test[counter]=classes.index(c)
    counter+=1

#x_test=x_test.reshape(test_size,224,224,3)/255
x_test=x_test.reshape((test_size,len(x_test)//test_size))
y_test=y_test.astype(np.int32)


slp=mlp.MLP([1024, 12])

epochs=500
batch_size=50
steps=len(x_train)//batch_size #Automatically adjust steps so that steps*batch_size is almost the number of samples
train_accs=[]
test_accs=[]
ep_vec=[]
plt.ion()
for i in range(epochs):
        slp.train(x_train, y_train, lr=0.001, lambda_=1e-5, momentum=0.99,
                      steps=steps, batch=batch_size)
        train_labels=slp.inference(x_train)[0]
        train_acc=(train_labels==y_train).mean()*100
        train_accs.append(train_acc)
        test_labels=slp.inference(x_test)[0]
        test_acc=(test_labels==y_test).mean()*100
        test_accs.append(test_acc)
        ep_vec.append(i)
        plt.clf()
        plt.plot(ep_vec, train_accs, label="Training Accuracy")
        plt.plot(ep_vec, test_accs, label="Test Accuracy")
        plt.pause(0.005)
plt.ioff()
if aug==False:
    slp.save("slp_params.npz")
else:
    slp.save("slp_aug_params.npz")
test_labels=slp.inference(x_test)[0]
acc=(test_labels==y_test).mean()*100
print("Single Layer Perceptron Test Accuracy (with PVMLNet as feature extractor): ", acc)
