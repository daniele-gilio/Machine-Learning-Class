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
#bow=pvml.maxabs_normalization(bow)
for i in range(len(t_pub_enc)):
    x_train[i][:-1]=bow[i]
    x_train[i][-1]=t_pub_enc[i]

######## Validation Set ####
######## Create BoW ########
bow=df.build_bow(val_title, False, "val_bow", size, stemming=stem)
val_pub_enc=np.array(val_pub_enc)


######## Consolidate Features ########
x_val=np.zeros((len(val_pub_enc), size+1))
#bow=pvml.maxabs_normalization(bow)
for i in range(len(val_pub_enc)):
    x_val[i][:-1]=bow[i]
    x_val[i][-1]=val_pub_enc[i]

######## Test Set ##########
######## Create BoW ########
bow=df.build_bow(test_title, False, "test_bow", size, stemming=stem)
test_pub_enc=np.array(test_pub_enc)


######## Consolidate Features ########
x_test=np.zeros((len(test_pub_enc), size+1))
#bow=pvml.maxabs_normalization(bow)

for i in range(len(test_pub_enc)):
    x_test[i][:-1]=bow[i]
    x_test[i][-1]=test_pub_enc[i]

######## Train a Multinomial Logistic Regression Model ###########
print("Multinomial Logistic Regression")
steps=10000
use_gpu=True
if use_gpu==True:
    import gpu_multi_logreg as gml
    start=time.time()
    w,b = gml.multinomial_logreg_train(x_train, y_train, 1e-5, lr=1e-2, steps=steps)
else:
    w,b = pvml.multinomial_logreg_train(x_train, y_train, 1e-5, lr=1e-2, steps=steps)

print("Total Time: ", time.time()-start)

np.savez("logreg_"+str(size), weights=w, biases=b)
scores=pvml.multinomial_logreg_inference(x_train, w, b)
labels=np.argmax(scores, axis=1)
acc=np.array(labels==y_train).mean()*100
print("Training Accuracy: ", acc)

scores=pvml.multinomial_logreg_inference(x_val, w, b)
labels=np.argmax(scores, axis=1)
acc=np.array(labels==y_val).mean()*100
print("Validation Accuracy: ", acc)

scores=pvml.multinomial_logreg_inference(x_test, w, b)
labels=np.argmax(scores, axis=1)
acc=np.array(labels==y_test).mean()*100
print("Test Accuracy: ", acc)

print("Elapsed Time: ", time.time()-start, "s")

conf_mat=np.zeros((4,4))
for i in range(len(y_test)):
    conf_mat[int(y_test[i])][int(labels[i])]+=1
for i in range(4):
    conf_mat[i,:]/=conf_mat[i,:].sum()

sn.heatmap(conf_mat, annot=True, cmap="coolwarm")
plt.title("Multinomial LogReg Confusion Matrix")
plt.show()
