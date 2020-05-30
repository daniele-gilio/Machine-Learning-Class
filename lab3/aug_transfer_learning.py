from pvml import pvmlnet, cnn, mlp
import glob
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt

def file_list(path):
    files = [f for f in glob.glob(path + "/" + "**/*.jpg", recursive=True)]
    return files

classes=["bluebell", "buttercup", "colts-foot", "daisy", "dandelion", "fritillary",
            "iris", "lily-valley", "pansy", "sunflower", "tigerlily", "windflower"]
train_path="flowers/train"
test_path="flowers/test"
n_c=len(classes)

train_file=file_list(train_path)
test_file=file_list(test_path)

train_size=60*12
test_size=20*12

x_train=image.imread(train_file[0])
y_train=np.zeros(train_size)
for c in classes:
    if c in train_file[0]:
        y_train[0]=classes.index(c)
x_test=image.imread(test_file[0])
y_test=np.zeros(test_size)
for c in classes:
    if c in test_file[0]:
        y_test[0]=classes.index(c)
counter=1

print("Loading Training Files...")
for f in train_file[1:]:
    x_train=np.append(x_train, image.imread(f))
    for c in classes:
        if c in f:
            y_train[counter]=classes.index(c)
    counter+=1

x_train=x_train.reshape(train_size,224,224,3)/255
y_train=y_train.astype(np.int32)

ind=np.arange(train_size)
np.random.shuffle(ind)
keep=30
x_t_keep=np.zeros((keep, 224,224,3))
y_t_keep=np.zeros(keep)
x_t_keep_2=np.zeros((keep, 224,224,3))
y_t_keep_2=np.zeros(keep)

for i in range(keep):
    x_t_keep[i]=x_train[ind[i]]
    y_t_keep[i]=y_train[ind[i]]
for i in range(keep):
    x_t_keep_2[i]=x_train[ind[i+keep]]
    y_t_keep_2[i]=y_train[ind[i+keep]]

y_t_keep=y_t_keep.astype(np.int32)
y_t_keep_2=y_t_keep_2.astype(np.int32)

counter=1

print("Loading Test Files...")
for f in test_file[1:]:
    x_test=np.append(x_test, image.imread(f))
    for c in classes:
        if c in f:
            y_test[counter]=classes.index(c)
    counter+=1

x_test=x_test.reshape(test_size,224,224,3)/255
y_test=y_test.astype(np.int32)

pvml_cnn=pvmlnet.PVMLNet.load("pvmlnet.npz")
slp=mlp.MLP.load("slp_params.npz")

pvml_cnn.weights[-1]=slp.weights[0][None, None, :, :]
pvml_cnn.biases[-1]=slp.biases[0]

pvml_cnn.update_w[-1]= np.zeros_like(pvml_cnn.weights[-1])
pvml_cnn.update_b[-1]= np.zeros_like(pvml_cnn.biases[-1])

test_labels=pvml_cnn.inference(x_test)[0]
p_acc=np.array((test_labels==y_test)).mean()*100
print("Pre Fine Tunining Network Accuracy: ", p_acc)

epochs=25
batch_size=4
steps=len(x_t_keep)//batch_size #Automatically adjust steps so that steps*batch_size is almost the number of samples
train_accs=[]
test_accs=[]
ep_vec=[]
lr0=1e-4
plt.ion()
for i in range(epochs):
    lr=lr0/np.sqrt(i+1)
    if epochs%2==0:
        pvml_cnn.train(x_t_keep, y_t_keep, lr=lr, lambda_=1e-5, momentum=0.99,
                      steps=steps, batch=batch_size)
        train_labels=pvml_cnn.inference(x_t_keep)[0]
        train_acc=np.array((train_labels==y_t_keep)).mean()*100
        train_accs.append(train_acc)
        test_labels=pvml_cnn.inference(x_test)[0]
        test_acc=np.array((test_labels==y_test)).mean()*100
        test_accs.append(test_acc)
        print(train_acc, p_acc-test_acc)
        ep_vec.append(i)
        plt.clf()
        plt.plot(ep_vec, train_accs, label="Training Accuracy")
        plt.plot(ep_vec, test_accs, label="Test Accuracy")
        plt.pause(0.005)
    else:
        pvml_cnn.train(x_t_keep_2, y_t_keep_2, lr=lr, lambda_=1e-5, momentum=0.99,
                      steps=steps, batch=batch_size)
        train_labels=pvml_cnn.inference(x_t_keep_2)[0]
        train_acc=np.array((train_labels==y_t_keep_2)).mean()*100
        train_accs.append(train_acc)
        test_labels=pvml_cnn.inference(x_test)[0]
        test_acc=np.array((test_labels==y_test)).mean()*100
        test_accs.append(test_acc)
        print(train_acc, test_acc)
        ep_vec.append(i)
        plt.clf()
        plt.plot(ep_vec, train_accs, label="Training Accuracy")
        plt.plot(ep_vec, test_accs, label="Test Accuracy")
        plt.pause(0.005)

plt.ioff()
pvml_cnn.save("fine_tuned_pvmlnet.npz")
test_labels=pvml_cnn.inference(x_test)[0]
acc=np.array((test_labels==y_test)).mean()*100
print("Fine Tuned Network Accuracy: ", acc)