import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["savefig.format"]="png"
#plt.rcParams['figure.figsize'] = [16,9]
np.random.seed(0)


#Import Data
names=["Class", "Gender", "Age", "Sibling and Spouses", "Parents and Children" , "Fare", "Survivors"]
data = np.loadtxt("titanic-train.txt")
test_data = np.loadtxt("titanic-test.txt")
#print(test_data.shape)

for i in range(7):
    plt.title(names[i])
    plt.xlabel(names[i])
    plt.ylabel("Relative Frequency")
    plt.grid(1)
    plt.hist(data[:,i], weights=(np.zeros_like(data[:,i])+1./data[:,i].size)*100)
    #plt.show()
    plt.savefig("images/"+names[i])
    plt.clf()

cl=data[:,0]
sex=data[:,1]
surv=data[:,6]
a=np.zeros(3)
b=np.zeros(2)
for i in range(cl.size):
    if sex[i]==0 and surv[i]==1:
        b[0]+=1
    if sex[i]==1 and surv[i]==1:
        b[1]+=1
    if cl[i]==1 and surv[i]==1:
        a[0]+=1
    elif cl[i]==2 and surv[i]==1:
        a[1]+=1
    elif cl[i]==3 and surv[i]==1:
        a[2]+=1

labc=[1,2,3]
labs=["Male", "Female"]
a/=a.sum()/100
b*=100/b.sum()
plt.bar(labc,a)
plt.grid(1)
plt.title("Class Survival Empirical Probability")
plt.xlabel("Class")
plt.ylabel("Surviving Probability")
#plt.show()
plt.savefig("images/"+"class_emp")
plt.clf()

plt.bar(labs,b)
plt.grid(1)
plt.title("Gender Survival Empirical Probability")
plt.xlabel("Gender")
plt.ylabel("Surviving Probability")
#plt.show()
plt.savefig("images/"+"gend_emp")
plt.clf()

#Plot Data Correlation
for i in range(6):
    for j in range(i+1,6):
        #print(i,j)
        plt.title(names[i] + " vs. " + names[j])
        plt.xlabel(names[i])
        plt.ylabel(names[j])
        plt.scatter(data[:,i], data[:,j], c=data[:,6], alpha=0.5)
        plt.grid(1)
        #plt.show()
        plt.savefig("images/"+names[i] + " vs. " + names[j]+".png")
        plt.clf()


#Logistic Regression Functions

def sigmoid(z):
    return 1./(1.+np.exp(-z))

def logreg_inference(x,w,b):
    return sigmoid(x@w+b)

def cross_entropy(p,y):
    return (-y*np.log(p)-(1-y)*np.log(1-p)).mean()

def logreg_train(x, y, epochs, lr,l, x_t,y_t):
    s_loss=np.zeros(epochs)
    s_tloss=np.zeros(epochs)
    s_acc=np.zeros(epochs)
    s_tacc=np.zeros(epochs)
    m,n=x.shape
    w=np.zeros(n)
    b=0
    n=0
    print("Steps  Loss   Test_Loss Accuracy Test_Accuracy")
    for step in range(epochs):
        p=logreg_inference(x,w,b)
        s_loss[step]=cross_entropy(p,y)
        predictions=np.asarray(p>0.5)
        s_acc[step]=np.asarray(predictions==y).mean()
        pt=logreg_inference(x_t,w,b)
        tpred=np.asarray(pt>0.5)
        s_tloss[step]= cross_entropy(pt,y_t)
        s_tacc[step]= np.asarray(tpred==y_t).mean()
        if step%10000==0:
            print(step, s_loss[step], s_tloss[step], s_acc[step]*100, s_tacc[step]*100)
        grad_b=(p-y).mean()#+2*l*(b)
        grad_w=(x.T@(p-y))/m+2*l*(w)
        b-=lr*grad_b
        w-=lr*grad_w

    np.save("loss", s_loss)
    np.save("loss_test", s_tloss)
    np.save("accuracy", s_acc)
    np.save("accuracy_test",s_tacc)
    return w,b

#Assign Data and Labels
x=data[:,:6]
y=data[:,6]
x_t=test_data[:,:6]
y_t=test_data[:,6]



#Training
w,b = logreg_train(x,y, int(2*1e5),0.001,0., x_t,y_t)
p=logreg_inference(x,w,b)
predictions=(p>0.5)
accuracy=(predictions==y).mean()
print("Training Set Accuracy: ", accuracy*100)



s_loss=np.load("loss.npy")
s_tloss=np.load("loss_test.npy")
s_acc=np.load("accuracy.npy")
s_tacc=np.load("accuracy_test.npy")

plt.plot(s_loss, label="Training Loss")
plt.plot(s_tloss, label="Test Loss")
plt.grid(1)
plt.legend()
plt.title("Loss Functions")
plt.xlabel("Steps")
plt.ylabel("Loss Value")
plt.savefig("images/"+"losses")
plt.clf()
#plt.show()

plt.plot(s_acc, label="Training Accuracy")
plt.plot(s_tacc, label="Test Accuracy")
plt.grid(1)
plt.legend()
plt.title("Accuracy")
plt.xlabel("Steps")
plt.ylabel("Accuracy Value")
plt.savefig("images/"+"acc")
plt.clf()
#plt.show()


#Results on the Training Set
print(w,b)

for i in range(6):
    for j in range(i+1,6):
        #print(i,j)
        plt.title(names[i] + " vs. " + names[j])
        plt.xlabel(names[i])
        plt.ylabel(names[j])
        decision_bound=-w[i]/w[j]*data[:,i]-b/w[j]
        plt.plot(data[:,i], decision_bound, label="Decision Bound")
        plt.scatter(data[:,i], data[:,j], c=data[:,6], alpha=0.5, label="Data")
        plt.savefig("images/"+"bound_"+ names[i] + " vs. " + names[j]+".png")
        plt.clf()
        #plt.show()


#Guess a Survival Probability
guess_data=np.zeros((1,6))

for i in range(5):
    guess_data[0][i]=np.random.randint(data[:,i].min(), data[:,i].max())

guess_data[0][5]=(data[:,5].max()-data[:,5].min())*np.random.rand()+data[:,5].min()
#print(guess_data)

prob_guess=logreg_inference(guess_data,w,b)
print("Random Guess: ", prob_guess*100)

#Educated Guess

guess_data[0][0]=1
guess_data[0][1]=1
edn=int(1e5)
ed_g=np.zeros(edn)
ed_d=np.zeros(edn)

for i in range(edn):
    guess_data[0][0]=1
    guess_data[0][1]=1
    for j in range(2,5):
        guess_data[0][j]=np.random.randint(data[:,j].min(), data[:,j].max())
    guess_data[0][5]=(data[:,5].max()-data[:,5].min())*np.random.rand()+data[:,5].min()
    guess_data[0][3]=1
    ed_g[i]=logreg_inference(guess_data,w,b)
    guess_data[0][0]=3
    guess_data[0][1]=0
    guess_data[0][3]=8
    ed_d[i]=logreg_inference(guess_data,w,b)

bins=np.zeros(10)
bind=np.zeros(10)
for i in range(edn):
    for j in range(10):
        if ed_g[i]*100<(j+1)*10 and ed_g[i]*100>j*10:
            bins[j]+=1
        if ed_d[i]*100<(j+1)*10 and ed_d[i]*100>j*10:
            bind[j]+=1

pl=np.zeros(10)
for i in range(10):
    pl[i]=(10*i+5)
print(pl)
plt.bar(pl,bins/bins.sum(), width=8, label="Women, Alone, 1 Class")
plt.bar(pl,bind/bind.sum(), width=8, label="Man, Not Alone, 3 Class")
plt.title(str(edn)+ " Educated Guesses")
plt.legend()
plt.grid()
plt.xticks(list(pl), pl)
plt.xlabel("Survival Probability")
plt.ylabel("Educated Guesses Percentage")
plt.savefig("images/"+"ed_guess")
plt.clf()
#plt.show()
prob_guess=logreg_inference(guess_data,w,b)
print("Educated Guess: ",prob_guess*100)
print(guess_data)

#My personal guess

guess_data[0][0]=3 #Class
guess_data[0][1]=0 #Gender
guess_data[0][2]=22 #Age
guess_data[0][2]=1 #Siblings/Spouses
guess_data[0][4]=2 #Parents/Children
guess_data[0][5]=20 #Fare

prob_guess=logreg_inference(guess_data,w,b)
print("My Probability (Worst Case): ", prob_guess*100.)

guess_data[0][0]=1 #Class
guess_data[0][5]=300 #Fare
prob_guess=logreg_inference(guess_data,w,b)
print("My Probability (Best Case): ", prob_guess*100.)

#Evaluate the model on test data
p=logreg_inference(x_t,w,b)
predictions=(p>0.5)
accuracy=(predictions==y_t).mean()
print("Test Accuracy: ", accuracy*100)

for i in range(6):
    for j in range(i+1,6):
        #print(i,j)
        plt.title(names[i] + " vs. " + names[j])
        plt.xlabel(names[i])
        plt.ylabel(names[j])
        decision_bound=-w[i]/w[j]*test_data[:,i]-b/w[j]
        plt.plot(test_data[:,i], decision_bound, label="Decision Bound")
        plt.scatter(test_data[:,i], test_data[:,j], c=test_data[:,6], alpha=0.5, label="Data")
        plt.grid(1)
        plt.savefig("images/"+"test_bound_"+names[i] + " vs. " + names[j]+".png")
        plt.clf()
        #plt.show()
