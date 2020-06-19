import gzip
from collections import Counter
import dictionary_functions as df
import pvml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import time

start=time.time()
####### Read Files ##########
f=gzip.open("train.txt.gz", "rt")
klass=[]
publisher=[]
title=[]
for line in f:
    k, p, t = line.split("|")
    klass.append(k)
    publisher.append(p)
    title.append(t)

####### Encode Classes and Publishers #########
pub=sorted(list(Counter(publisher).keys())) #Get the list of unique publishers
kla=sorted(list(Counter(klass).keys())) #Get the list of unique classes

pub_enc=[]
klass_enc=[]
for p in publisher:
    pub_enc.append(pub.index(p)) #use the publisher index in pub to encode publishers
for k in klass:
    klass_enc.append(kla.index(k)) #use the class index in kla to encode classes

y_train=np.array(klass_enc)
t_pub_enc=pub_enc
t_title=title

######## Repeat for Validation and Test Sets #########
####### Read Files ##########
f=gzip.open("validation.txt.gz", "rt")
klass=[]
publisher=[]
title=[]
for line in f:
    k, p, t = line.split("|")
    klass.append(k)
    publisher.append(p)
    title.append(t)

####### Encode Classes and Publishers #########
pub=sorted(list(Counter(publisher).keys())) #Get the list of unique publishers
kla=sorted(list(Counter(klass).keys())) #Get the list of unique classes
pub_enc=[]
klass_enc=[]
for p in publisher:
    pub_enc.append(pub.index(p)) #use the publisher index in pub to encode publishers
for k in klass:
    klass_enc.append(kla.index(k)) #use the class index in kla to encode classes

y_val=np.array(klass_enc)
val_pub_enc=pub_enc
val_title=title

####### Read Files ##########
f=gzip.open("test.txt.gz", "rt")
klass=[]
publisher=[]
title=[]
for line in f:
    k, p, t = line.split("|")
    klass.append(k)
    publisher.append(p)
    title.append(t)

####### Encode Classes and Publishers #########
pub=sorted(list(Counter(publisher).keys())) #Get the list of unique publishers
kla=sorted(list(Counter(klass).keys())) #Get the list of unique classes

pub_enc=[]
klass_enc=[]
for p in publisher:
    pub_enc.append(pub.index(p)) #use the publisher index in pub to encode publishers
for k in klass:
    klass_enc.append(kla.index(k)) #use the class index in kla to encode classes

y_test=np.array(klass_enc)
test_pub_enc=pub_enc
test_title=title



######## Create Dictionary #########
size=8000
stem=True
ic=True
dic=df.build_dict(t_title, size, True, stemming=stem, ignore_common=ic)

######## Training Set ######
######## Create BoW ########
bow=df.build_bow(t_title, False, "train_bow", size, stemming=stem)
t_pub_enc=np.array(t_pub_enc)


######## Consolidate Features ########
x_train=np.zeros((len(t_pub_enc), size+1))
bow=pvml.maxabs_normalization(bow)
for i in range(len(t_pub_enc)):
    x_train[i][:-1]=bow[i]
    x_train[i][-1]=t_pub_enc[i]

######## Validation Set ####
######## Create BoW ########
bow=df.build_bow(val_title, False, "val_bow", size, stemming=stem)
val_pub_enc=np.array(val_pub_enc)


######## Consolidate Features ########
x_val=np.zeros((len(val_pub_enc), size+1))
bow=pvml.maxabs_normalization(bow)
for i in range(len(val_pub_enc)):
    x_val[i][:-1]=bow[i]
    x_val[i][-1]=val_pub_enc[i]

######## Test Set ##########
######## Create BoW ########
bow=df.build_bow(test_title, False, "test_bow", size, stemming=stem)
test_pub_enc=np.array(test_pub_enc)


######## Consolidate Features ########
x_test=np.zeros((len(test_pub_enc), size+1))
bow=pvml.maxabs_normalization(bow)

for i in range(len(test_pub_enc)):
    x_test[i][:-1]=bow[i]
    x_test[i][-1]=test_pub_enc[i]

######## Train a Multi Layer Perceptron #########
print("Multi-Layer Perceptron")
dnn=pvml.MLP([size+1, 256, 4]) #16

epochs=375 #<- Starts overfitting slightly if we go over that number of epochs
batch_size=256
steps=len(x_train)//batch_size #Automatically adjust steps so that steps*batch_size is almost the number of samples
train_accs=[]
val_accs=[]
ep_vec=[]
plt.ion()
lr0=0.01
for i in range(epochs):
        lr=lr0/np.sqrt(i+1)
        dnn.train(x_train, y_train, lr=lr, lambda_=1e-5, momentum=0.99,
                      steps=steps, batch=batch_size)
        train_labels=dnn.inference(x_train)[0]
        train_acc=(train_labels==y_train).mean()*100
        train_accs.append(train_acc)
        val_labels=dnn.inference(x_val)[0]
        val_acc=(val_labels==y_val).mean()*100
        val_accs.append(val_acc)
        ep_vec.append(i)
        plt.clf()
        plt.plot(ep_vec, train_accs, label="Training Accuracy")
        plt.plot(ep_vec, val_accs, label="Validation Accuracy")
        plt.legend()
        plt.grid(1)
        plt.pause(0.005)
plt.ioff()

dnn.save("dnn_"+str(size))

plt.clf()
plt.plot(ep_vec, train_accs, label="Training")
plt.plot(ep_vec, val_accs, label="Validation")
plt.title("MLP Training")
plt.xlabel("Epochs")
plt.ylabel("Accuracy [%]")
plt.legend()
plt.grid(1)
plt.savefig("mlp_training.png")

print("Final Training Accuracy: ", train_accs[-1])
print("Final Validation Accuracy: ", val_accs[-1])
test_labels=dnn.inference(x_test)[0]
test_acc=(test_labels==y_test).mean()*100
print("Test Accuracy: ", test_acc)

print("Elapsed Time: ", time.time()-start, "s")

conf_mat=np.zeros((4,4))
for i in range(len(y_test)):
    conf_mat[int(y_test[i])][int(test_labels[i])]+=1
for i in range(4):
    conf_mat[i,:]/=conf_mat[i,:].sum()

sn.heatmap(conf_mat, annot=True, cmap="coolwarm")
plt.title("MLP Confusion Matrix")
plt.show()
