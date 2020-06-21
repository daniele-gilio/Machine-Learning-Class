import gzip
from collections import Counter
import dictionary_functions as df
import pvml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

print("Creating useful information for results analysis...")
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

print("Done reading the Dataset")
print()

q=open("final_results.txt", "w")
print("Naive Bayes Classifier")
print("Naive Bayes Classifier", file=q)
######## Bayes #########
data=np.load("bayes_"+str(size)+".npz", allow_pickle=True)
w=data["weights"]
b=data["biases"]
labels, scores = pvml.multinomial_naive_bayes_inference(x_train, w, b)
acc=(labels==y_train).mean()*100
print("Training Accuracy: ", acc)
print("Training Accuracy: ", acc, file=q)

labels, scores = pvml.multinomial_naive_bayes_inference(x_val, w, b)
acc=(labels==y_val).mean()*100
print("Validation Accuracy: ", acc)
print("Validation Accuracy: ", acc, file=q)

labels, scores = pvml.multinomial_naive_bayes_inference(x_test, w, b)
acc=(labels==y_test).mean()*100
print("Test Accuracy: ", acc)
print("Test Accuracy: ", acc, file=q)

b_acc=acc

w_n=5 #<- Number of Wrong Classifications to show
wrong=np.asarray(labels!=y_test)

w_scores=[]
for i in range(len(wrong)):
    if wrong[i]==True:
        w_scores.append(scores[i][labels[i]])
    else:
        w_scores.append(-5000)

w_scores=np.array(w_scores)
w_index=w_scores.argsort()

f=open("wrong_bayes.txt", "w")
print("Wrongest Classifications", file=f)
print(file=f)
for i in range(1,w_n+1):
    print("Title:", test_title[w_index[-i]], file=f)
    print("Correct Class:", kla[y_test[w_index[-i]]], "Predicted Class:", kla[labels[w_index[-i]]], file=f)
    print(file=f)

f.close()

conf_mat=np.zeros((4,4))
for i in range(len(y_test)):
    conf_mat[int(y_test[i])][int(labels[i])]+=1
for i in range(4):
    conf_mat[i,:]/=conf_mat[i,:].sum()

sn.heatmap(conf_mat, annot=True, cmap="coolwarm")
plt.title("Multinomial Naive Bayes Confusion Matrix")
plt.xticks(np.linspace(0.5, 3.5, 4),kla)
plt.yticks(np.linspace(0.5, 3.5, 4),kla, rotation="horizontal")
plt.savefig("bayes_confmat.png")
plt.clf()


bayes_acc=[82.3, 88.3, 90.4, 90.3, 90.0] #no-norm
#2500
#88.3 no-norm
#85.7 maxabs
#87.9 ignore_common
#88.0 stemming
#88.9 ignore_common+stemming
print("Naive Bayes Classifier Done")
print("Naive Bayes Classifier Done", file=q)
print()
print(file=q)
print()
print(file=q)


print("Logistic Regression")
print("Logistic Regression", file=q)
######## LogReg ############
data=np.load("logreg_"+str(size)+".npz", allow_pickle=True)
w=data["weights"]
b=data["biases"]
scores=pvml.multinomial_logreg_inference(x_train, w, b)
labels=np.argmax(scores, axis=1)
acc=np.array(labels==y_train).mean()*100
print("Training Accuracy: ", acc)
print("Training Accuracy: ", acc, file=q)

scores=pvml.multinomial_logreg_inference(x_val, w, b)
labels=np.argmax(scores, axis=1)
acc=np.array(labels==y_val).mean()*100
print("Validation Accuracy: ", acc)
print("Validation Accuracy: ", acc, file=q)

scores=pvml.multinomial_logreg_inference(x_test, w, b)
labels=np.argmax(scores, axis=1)
acc=np.array(labels==y_test).mean()*100
print("Test Accuracy: ", acc)
print("Test Accuracy: ", acc, file=q)

l_acc=acc

wrong=np.asarray((labels!=y_test))

w_scores=[]
for i in range(len(wrong)):
    if wrong[i]==True:
        w_scores.append(scores[i][labels[i]])
    else:
        w_scores.append(-1)

w_scores=np.array(w_scores)
w_index=w_scores.argsort()


f=open("wrong_logreg.txt", "w")
print("Wrongest Classifications", file=f)
print(file=f)
for i in range(1,w_n+1):
    print("Title:", test_title[w_index[-i]], file=f)
    print("Correct Class:", kla[y_test[w_index[-i]]], "Predicted Class:", kla[labels[w_index[-i]]], file=f)
    print(file=f)

f.close()

conf_mat=np.zeros((4,4))
for i in range(len(y_test)):
    conf_mat[int(y_test[i])][int(labels[i])]+=1
for i in range(4):
    conf_mat[i,:]/=conf_mat[i,:].sum()

sn.heatmap(conf_mat, annot=True, cmap="coolwarm")
plt.title("Multinomial LogReg Confusion Matrix")
plt.xticks(np.linspace(0.5, 3.5, 4),kla)
plt.yticks(np.linspace(0.5, 3.5, 4),kla, rotation="horizontal")
plt.savefig("logreg_confmat.png")
plt.clf()

logreg_acc=[72.8, 75.1, 75.6, 75.5, 75.6]
#2500
#75.1 no-norm
#72.6 maxabs
#74.4 ignore_common
#76.8 stemming
#76.7 ignore_common+stemming

print("Done with Logreg")
print("Done with Logreg", file=q)
print()
print(file=q)
print()
print(file=q)

########### Normalize Bow with MaxAbs for SLP and MLP (as in training) ############
for i in range(len(t_pub_enc)):
    x_train[i][:-1]=pvml.maxabs_normalization(x_train[i][:-1])
for i in range(len(val_pub_enc)):
    x_val[i][:-1]=pvml.maxabs_normalization(x_val[i][:-1])
for i in range(len(test_pub_enc)):
    x_test[i][:-1]=pvml.maxabs_normalization(x_test[i][:-1])

print("Single Layer Perceptron")
print("Single Layer Perceptron", file=q)
########## SLP ##########
snn=pvml.MLP.load("snn_"+str(size)+".npz")
train_labels=snn.inference(x_train)[0]
train_acc=(train_labels==y_train).mean()*100
print("Final Training Accuracy: ", train_acc)
print("Final Training Accuracy: ", train_acc, file=q)
val_labels=snn.inference(x_val)[0]
val_acc=(val_labels==y_val).mean()*100
print("Final Validation Accuracy: ", val_acc)
print("Final Validation Accuracy: ", val_acc, file=q)
test_labels, test_scores=snn.inference(x_test)
test_acc=(test_labels==y_test).mean()*100
print("Test Accuracy: ", test_acc)
print("Test Accuracy: ", test_acc, file=q)

s_acc=test_acc

wrong=np.asarray((test_labels!=y_test))

w_scores=[]
for i in range(len(wrong)):
    if wrong[i]==True:
        w_scores.append(test_scores[i][test_labels[i]])
    else:
        w_scores.append(-1)

w_scores=np.array(w_scores)
w_index=w_scores.argsort()


f=open("wrong_slp.txt", "w")
print("Wrongest Classifications", file=f)
print(file=f)
for i in range(1,w_n+1):
    print("Title:", test_title[w_index[-i]], file=f)
    print("Correct Class:", kla[y_test[w_index[-i]]], "Predicted Class:", kla[test_labels[w_index[-i]]], file=f)
    print(file=f)

f.close()

conf_mat=np.zeros((4,4))
for i in range(len(y_test)):
    conf_mat[int(y_test[i])][int(test_labels[i])]+=1
for i in range(4):
    conf_mat[i,:]/=conf_mat[i,:].sum()

sn.heatmap(conf_mat, annot=True, cmap="coolwarm")
plt.title("SLP Confusion Matrix")
plt.xticks(np.linspace(0.5, 3.5, 4),kla)
plt.yticks(np.linspace(0.5, 3.5, 4),kla, rotation="horizontal")
plt.savefig("slp_confmat.png")
plt.clf()

slp_acc=[83.2, 87.6, 89.6, 89.9, 90.2]
#2500
#87.6 no-norm
#87.9 maxabs
#88.2 ignore_common
#89.4 Stemming
#89.4 ignore_common+stemming

print("Done with SLP")
print("Done with SLP", file=q)
print()
print(file=q)
print()
print(file=q)

########### Accuracy vs Dictionary Size #########
plt.plot(np.linspace(1000, 10000, 5), bayes_acc, label="Bayes", marker=".")
plt.plot(np.linspace(1000, 10000, 5), logreg_acc, label="LogReg", marker=".")
plt.plot(np.linspace(1000, 10000, 5), slp_acc, label="SLP", marker=".")
plt.title("Accuracy Vs. Dictionary Size")
plt.xlabel("Dictionary Size")
plt.ylabel("Accuracy [%]")
plt.grid()
plt.legend()
plt.savefig("acc_vs_dic.png")
plt.clf()

########### MLP width tests #############
x=[8,16,32,64,128,256,512]
y=[88.1,91.2,90.5,90.7,90.3,91.1,91]
plt.plot(x,y, marker=".")
plt.semilogx()
plt.grid()
plt.title("Accuracy Vs. Hidden Layer Size")
plt.xlabel("Layer Width (Log)")
plt.ylabel("Test Accuracy [%]")
plt.savefig("mlp_tests.png")
plt.clf()


print("Multi-Layer Perceptron")
print("Multi-Layer Perceptron", file=q)
########## MLP ##########
dnn=pvml.MLP.load("dnn_"+str(size)+".npz")
train_labels=dnn.inference(x_train)[0]
train_acc=(train_labels==y_train).mean()*100
print("Final Training Accuracy: ", train_acc)
print("Final Training Accuracy: ", train_acc, file=q)
val_labels=dnn.inference(x_val)[0]
val_acc=(val_labels==y_val).mean()*100
print("Final Validation Accuracy: ", val_acc)
print("Final Validation Accuracy: ", val_acc, file=q)
test_labels, test_scores=dnn.inference(x_test)
test_acc=(test_labels==y_test).mean()*100
print("Test Accuracy: ", test_acc)
print("Test Accuracy: ", test_acc, file=q)
mlp_acc=test_acc

wrong=np.asarray((test_labels!=y_test))

w_scores=[]
for i in range(len(wrong)):
    if wrong[i]==True:
        w_scores.append(test_scores[i][test_labels[i]])
    else:
        w_scores.append(-1)

w_scores=np.array(w_scores)
w_index=w_scores.argsort()


f=open("wrong_mlp.txt", "w")
print("Wrongest Classifications", file=f)
print(file=f)
for i in range(1,w_n+1):
    print("Title:", test_title[w_index[-i]], file=f)
    print("Correct Class:", kla[y_test[w_index[-i]]], "Predicted Class:", kla[test_labels[w_index[-i]]], file=f)
    print(file=f)

f.close()

conf_mat=np.zeros((4,4))
for i in range(len(y_test)):
    conf_mat[int(y_test[i])][int(test_labels[i])]+=1
for i in range(4):
    conf_mat[i,:]/=conf_mat[i,:].sum()

sn.heatmap(conf_mat, annot=True, cmap="coolwarm")
plt.title("MLP Confusion Matrix")
plt.xticks(np.linspace(0.5, 3.5, 4),kla)
plt.yticks(np.linspace(0.5, 3.5, 4),kla, rotation="horizontal")
plt.savefig("mlp_confmat.png")
plt.clf()
#[size+1, 256, 64, 16, 4] (ic+stem)(2500) 89.7
#[size+1, 8, 8, 4] // // 84.89
#[size+1, 64, 8, 4] // // 89.6
#[size+1, 32, 8, 4] // // 87.2
#[size+1, 512, 64, 8, 4] // // 87.9
#[size+1, 256, 64, 8, 4] // // 87.1
#[size+1, 512, 4] // // 91.0
#[size+1, 256, 4] // // 91.1
#[size+1, 128, 4] // // 90.3
#[size+1, 64, 4] // // 90.7
#[size+1, 32, 4] // // 90.5
#[size+1, 16, 4] // // 91.2
#[size+1, 8, 4] // // 88.1

print("MLP Done")
print("MLP Done", file=q)
print()
print(file=q)
print()
print(file=q)

q.close()

print("Creating Graphs...")
########### Accuracy vs. Training Time #############
bayes_time=2.774
logreg_time=912.641
slp_time=338.017
mlp_time=646.121
_time=[bayes_time, logreg_time, slp_time, mlp_time]
_acc=[b_acc, l_acc, s_acc, mlp_acc]
sz=[100, 10000, 1500, 350] #number of epochs to train the models, the bayes number is arbitrary but smaller than
                            #all the others since it does not use Gradient Descent
names=["Bayes", "LogReg", "SLP", "MLP"]
plt.rcParams['figure.figsize'] = 16,9

for i in range(4):
    plt.scatter(_time[i], _acc[i], s=sz[i], label=names[i])

plt.legend(markerscale=0.18)
plt.title("Accuracy Vs. Training Time")
plt.xlabel("Time [s]")
plt.ylabel("Accuracy [%]")
plt.grid()
plt.savefig("acc_vs_time.png")
plt.clf()


######## Most Polarizing Words #########
f=open("most_polarizing.txt", "w")
dic=df.get_dict(size)
for j in range(19):
    diag=np.zeros((size+1,size+1))
    for i in range(size+1):
        diag[i][i]=1
        diag[i][-1]=j
    labels, scores=dnn.inference(diag)
    h=scores[0][labels[0]]
    l=scores[0][labels[0]]
    h_ind=0
    l_ind=0
    print("\033[1m" + "Publisher: ", pub[j], '\033[0m', file=f)
    for t in range(4):
        for k in range(1,size+1):
            if scores[k][labels[k]]>h and labels[k]==t:
                h=scores[k][labels[k]]
                h_ind=k
            elif scores[k][labels[k]]<l and labels[k]==t:
                l=scores[k][labels[k]]
                l_ind=k
        print("\033[1m", kla[t], '\033[0m', file=f)
        print("Most Influencial Word:", df.get_word(dic, h_ind), ", Score:", h, file=f)
        print("Least Influencial Word:", df.get_word(dic, l_ind), ", Score:", l, file=f)
        print(file=f)
    print(file=f)

print("Done!")
