import numpy as np 
import matplotlib.pyplot as plt 

def sidmoind(number):
    return (1/(1+np.exp(-number)))

class_1 = np.random.multivariate_normal(mean =(1,1), cov =[[0.5, 0], [0, 0.5]], size= 100)
class_2 = np.random.multivariate_normal(mean =(-1,-1), cov =[[0.5, 0], [0, 0.5]], size= 100)

label_1 = np.ones(shape=100)
label_2 = np.zeros(shape=100)

dataset = np.vstack((class_1, class_2))
lables = np.concatenate((label_1,label_2))
lables = np.transpose(lables)

J = 0 
b = 0
W = np.array([0,0])
dW = np.array([0,0])
X = np.transpose(dataset)
m=200
q = 0.001

for i in range(1000):
    z = np.dot(W,X) + b
    A = sidmoind(z)
    J += -(np.dot(lables , np.log10(A)) + np.dot((1-lables),np.log10(1-A)))/200
    dz = A - lables
    dW = np.dot(X,np.transpose(dz))/m
    db = np.sum(dz)/m
    W = W - q * dW
    b = b - q * db
#print(W, b)

new = [-(W[1]*dataset[i,1]+ b)/W[0] for i in range(200)]

plt.scatter(x = dataset[:,0],y=dataset[:,1],c=lables)
plt.plot(new,dataset[:,1])
plt.show()