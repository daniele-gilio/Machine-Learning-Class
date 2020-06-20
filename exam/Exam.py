import gzip
from collections import Counter
import dictionary_functions as df

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
pub=list(Counter(publisher).keys()) #Get the list of unique publishers
kla=list(Counter(klass).keys()) #Get the list of unique classes

pub_enc=[]
klass_enc=[]
for p in publisher:
    pub_enc.append(pub.index(p)) #use the publisher index in pub to encode publishers
for k in klass:
    klass_enc.append(kla.index(k)) #use the class index in kla to encode classes

######## Create Dictionary #########
dic=df.build_dict(title, 100, False)
print(dic)

#for i in range(len(klass)):
    #print(klass[i], publisher[i], title[i])
