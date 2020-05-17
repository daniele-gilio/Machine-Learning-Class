from pvml import naivebayes as nb
import numpy as np
from collections import Counter as counter
import glob
import pandas as pd #we use pandas to read files, it is way faster than numpy with big files

def file_list(path, pn): #pn stands for positive/negative
    files = [f for f in glob.glob(path + "/" + pn + "**/*.txt", recursive=True)]
    return files

def most_influential(w,n):
    index=np.argmax(w, 1)
    pos=np.zeros(len(index))
    neg=np.zeros(len(index))
    diff=w[:,0]-w[:,1]
    for i in range(len(index)):
        if index[i]==1 and diff[i]<0:
            pos[i]=diff[i]
        else:
            neg[i]=diff[i]
    pos=abs(pos)
    neg=abs(neg)
    return pos.argsort()[-n:], neg.argsort()[-n:]

######Load Files#######
n = np.loadtxt("parameters.txt", unpack=True, usecols=(0))
n=int(n)
test_path="aclImdb/test"
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

#######################
########Train the Classifier#########

X=train_data[:,:-1]
Y=train_data[:,-1]

w,b=nb.multinomial_naive_bayes_train(X,Y)
labels,scores=nb.multinomial_naive_bayes_inference(X,w,b)

accuracy=(labels==Y).mean()*100
print("Training Accuracy: ", accuracy)


X=val_data[:,:-1]
Y=val_data[:,-1]
labels,scores=nb.multinomial_naive_bayes_inference(X,w,b)
accuracy=(labels==Y).mean()*100
print("Validation Accuracy: ", accuracy)

X=test_data[:,:-1]
Y=test_data[:,-1]
labels,scores=nb.multinomial_naive_bayes_inference(X,w,b)
accuracy=(labels==Y).mean()*100
print("Test Accuracy: ", accuracy)



mi=20 #most influential words number
pos, neg=most_influential(w,mi)

#Recreate Vocabulary
f=open("voc/vocabulary_" +str(n)+".txt", "r")
vo=[]

for l in f.read().split():
        vo.append(l)

f.close()
f=open("most_influential.txt","w")

for i in range(mi):
    if i==0:
        print("Best Words,   Worst Words")
        print("Best Words, Worst Words", file=f)
    print(vo[pos[mi-1-i]], "      ", vo[neg[mi-1-i]])#, vo[pos[mi-1-i]], vo[neg[mi-1-i]])
    print(vo[pos[mi-1-i]], " ", vo[neg[mi-1-i]], file=f)

f.close()

#########Compute Wrong Classifications#########

w_n=5 #<- Number of Wrong Classifications to show
wrong=np.asarray((labels!=Y))

w_scores=[]
for i in range(len(wrong)):
    if wrong[i]==True:
        w_scores.append(abs(scores[i,0]-scores[i,1]))

w_scores=np.array(w_scores)
w_index=w_scores.argsort()

f=open("wrong_classifications.txt", "w")
for i in range(w_n):
    print(file_list(test_path, "")[w_index[w_n-1-i]])
    print(file_list(test_path, "")[w_index[w_n-1-i]],file=f)
f.close()
