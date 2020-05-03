# Load required packages and dataset. Do not modify.
import matplotlib.pyplot as plt
import numpy as np

def load_iris_dataset():
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y
    
X, y = load_iris_dataset()

# Implement your solution here.
print("The size of the dataset is %d"%len(X))
setosa=versicolor=virginica=0
for i in y:
    if i==0:
        setosa+=1
    if i==1:
        versicolor+=1
    if i==2:
        virginica+=1
print("The Number of setosa is%d"%setosa)
print("The Number of versicolor is%d"%versicolor)
print("The Number of virginica is%d"%virginica)


meanSL=np.mean(X[:,0])
meanSW=np.mean(X[:,1])
meanPL=np.mean(X[:,2])
meanPW=np.mean(X[:,3])

stdSL=np.std(X[:,0])
stdSW=np.std(X[:,1])
stdPL=np.std(X[:,2])
stdPW=np.std(X[:,3])

print("The mean of SL is %f"%meanSL)
print("The mean of SW is %f"%meanSW)
print("The mean of PL is %f"%meanPL)
print("The mean of PW is %f"%meanPW)

print("The std of SL is %f"%stdSL)
print("The std of SW is %f"%stdSW)
print("The std of PL is %f"%stdPL)
print("The std of PW is %f"%stdPW)

# plt.scatter(X[:,0],X[:,2],c=y)
# plt.title('Visualize the variables SL und PL')
# plt.xlabel('Sepal Length')
# plt.ylabel('Petal Length')

# plt.show()


def train_test_split(X, y):
    """
    Returns X_train, X_test, y_train, y_test, 
        where X_train and X_test are the input features of the training and test set,
        and y_train and y_test are the class labels of the training and test set.
    """
    # Implement your solution here.
    c=np.c_[X,y] 
    # print(c)
    np.random.shuffle(c)
    X1=c[:,0:4]
    y1=c[:,4]
    y1=y1.astype(int)
    l=int(0.7*len(X))

    return X1[0:l,:], X1[l:len(X),:], y1[0:l], y1[l:y.size]

X_train, X_test, y_train, y_test = train_test_split(X, y)

assert (X_train.shape[0] + X_test.shape[0]) == X.shape[0]
assert (y_train.shape[0] + y_test.shape[0]) == y.shape[0]
assert X_train.shape[1] == X_test.shape[1]

vmin=X_train.min(axis=0)
vmax=X_train.max(axis=0)
print(vmin,vmax)

N_X_train=X_train
N_X_test=X_test
for i in range(4):
    N_X_train[:,i] = (X_train[:,i] - vmin[i]) / (vmax[i] - vmin[i])

for i in range(4):
    N_X_test[:,i] = (X_test[:,i] - vmin[i]) / (vmax[i] - vmin[i])

# N_X_train=X_train
# for i in range(0,4):
#     for j in range(len(X_train)):
#         N_X_train[j,i]=(X_train[j,i]-min(X_train[:,i]))/(max(X_train[:,i])-min(X_train[:,i]))

# N_X_test=X_test
# for i in range(0,4):
#     for j in range(len(X_test)):
#         N_X_test[j,i]=(X_test[j,i]-min(X_test[:,i]))/(max(X_test[:,i])-min(X_test[:,i]))
# print(N_X_train)
# print(y_train)
plt.scatter(N_X_train[:,0],N_X_train[:,2],c=y_train)
plt.title('Visualize the variables SL und PL')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.show()
def euclidean_distance(x1, x2):
    """
    Given vectors x1 and x2 with shape (n,) returns distance between vectors as float.
    """
    return np.sqrt(np.sum((x1 - x2)*(x1 - x2)))

class KNearestNeighbors(object):
    def __init__(self, k, distance_metric='uniform'):
        self.k = k
        self.distance_metric = distance_metric
        
    def fit(self, X, y):
        """
        This functions saves the training data to be used during the prediction.
        """
        self.X = X
        self.y = y
    
    def predict(self, X):
        """
        Returns a vector of shape (n,) if X has shape (n,d), 
        where n is the number of samples and d is the number of features.
        """
        # Implement your solution here.
        f0=f1=f2=0.0
        cnt0=cnt1=cnt2=0
        for i in range(self.y.size):
            if self.y[i]==0:
                f0+=1
            if self.y[i]==1:
                f1+=1
            if self.y[i]==2:
                f2+=1
        f0=f0/self.y.size
        f1=f1/self.y.size
        f2=f2/self.y.size

        result=np.zeros(shape=(X.shape[0],2))
        
        for i in range(X.shape[0]):
            array_dist=np.zeros(shape=(self.X.shape[0],2))
            for j in range(self.X.shape[0]):

                    dis= self.distance_metric(X[i,:],self.X[j,:])

                    array_dist[j,:]=dis,self.y[j]
            array_dist=array_dist[array_dist[:,0].argsort()]

            array_dist=array_dist[0:self.k]

            for k in range(len(array_dist)):
                if array_dist[k,1]==0.0:
                    cnt0+=1
                if array_dist[k,1]==1.0:
                    cnt1+=1
                if array_dist[k,1]==2.0:
                    cnt2+=1
            result[i,0]=i
            listWeight=[cnt0/f0,cnt1/f1,cnt2/f2]

            result[i,1]=listWeight.index(max(listWeight))
            cnt0=0
            cnt1=0
            cnt2=0
        return result
        



# def euclidean_distance(x1, x2):
#     """
#     Given vectors x1 and x2 with shape (n,) returns distance between vectors as float.
#     """
#     return np.sqrt(np.sum((x1 - x2)*(x1 - x2)))

def precision(y_pred, y_true):
    # Implement your solution here.
    a11=a22=a33=0

    totalOne_true=np.sum(y_true[:,1]==0)   
    totalTwo_true=np.sum(y_true[:,1]==1)
    totalThree_true=np.sum(y_true[:,1]==2)

    for i in range(len(y_pred)):
        if y_pred[i,1]==y_true[i,1]:
            if y_pred[i,1]==0:
                a11+=1
            if y_pred[i,1]==1:
                a22+=1
            if y_pred[i,1]==2:
                a33+=1
    P1=float(a11)/float(totalOne_true)
    P2=float(a22)/float(totalTwo_true)
    P3=float(a33)/float(totalThree_true)
    return P1,P2,P3,((P1+P2+P3)/3)

def recall(y_pred, y_true):
    # Implement your solution here.
    a11=a22=a33=0
    
    totalOne_pred=np.sum(y_pred[:,1]==0)
    totalTwo_pred=np.sum(y_pred[:,1]==1)
    totalThree_pred=np.sum(y_pred[:,1]==2)

    for i in range(len(y_pred)):
        if y_pred[i,1]==y_true[i,1]:
            if y_pred[i,1]==0:
                a11+=1
            if y_pred[i,1]==1:
                a22+=1
            if y_pred[i,1]==2:
                a33+=1
    R1=float(a11)/float(totalOne_pred)
    R2=float(a22)/float(totalTwo_pred)
    R3=float(a33)/float(totalThree_pred)
    return R1,R2,R3,((R1+R2+R3)/3)

def f1score(y_pred, y_true):
    # Implement your solution here.
    
    a11=a22=a33=0
    
    totalOne_pred=np.sum(y_pred[:,1]==0)
    totalTwo_pred=np.sum(y_pred[:,1]==1)
    totalThree_pred=np.sum(y_pred[:,1]==2)

    totalOne_true=np.sum(y_true[:,1]==0)   
    totalTwo_true=np.sum(y_true[:,1]==1)
    totalThree_true=np.sum(y_true[:,1]==2)

    for i in range(len(y_pred)):
        if y_pred[i,1]==y_true[i,1]:
            if y_pred[i,1]==0:
                a11+=1
            if y_pred[i,1]==1:
                a22+=1
            if y_pred[i,1]==2:
                a33+=1
    P1=float(a11)/float(totalOne_true)
    P2=float(a22)/float(totalTwo_true)
    P3=float(a33)/float(totalThree_true)
    R1=float(a11)/float(totalOne_pred)
    R2=float(a22)/float(totalTwo_pred)
    R3=float(a33)/float(totalThree_pred)

    F1=2*P1*R1/(P1+R1)
    F2=2*P2*R2/(P2+R2)
    F3=2*P3*R3/(P3+R3)
    return F1,F2,F3,((F1+F2+F3)/3)

def acc(y_pred, y_true):
    cnt=0
    for i in range(len(y_pred)):
        if y_pred[i,1]==y_true[i,1]:
            cnt+=1
    return float(cnt)/float(len(y_pred)) 

y_true=np.zeros(shape=(X_test.shape[0],2))
for i in range(y_test.size):
    y_true[i,1]=y_test[i]
    y_true[i,0]=i

listResult1=[]
listResult3=[]
listResult5=[]

knn3=KNearestNeighbors(3,distance_metric=euclidean_distance)
knn5=KNearestNeighbors(5,distance_metric=euclidean_distance)
knn1=KNearestNeighbors(1,distance_metric=euclidean_distance)

knn1.fit(N_X_train,y_train)
knn3.fit(N_X_train,y_train)
knn5.fit(N_X_train,y_train)

listResult1=knn1.predict(N_X_test)
listResult3=knn3.predict(N_X_test)
listResult5=knn5.predict(N_X_test)


Precision1=precision(listResult1,y_true) 
Precision3=precision(listResult3,y_true)
Precision5=precision(listResult5,y_true)

Recall1=recall(listResult1,y_true)
Recall3=recall(listResult3,y_true)
Recall5=recall(listResult5,y_true)

F11=f1score(listResult1,y_true)
F13=f1score(listResult3,y_true)
F15=f1score(listResult5,y_true)

acc1=acc(listResult1,y_true)
acc3=acc(listResult3,y_true)
acc5=acc(listResult5,y_true)

print(acc1,acc3,acc5)
print(Precision1,Precision3,Precision5)
print(Recall1,Recall3,Recall5)
print(F11,F13,F15)
