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




J=0
dw1=0
dw2=0
db=0
w1 = 0 
w2 = 0
b = 0
q = 0.001
for item in range(1000):
    for n in range(200):
     z = w1 * dataset[n,0] + w2 * dataset[n,1] + b
     a = sidmoind(z)
     J += -(lables[n] * np.log10(a) + (1-lables[n]) * np.log10(1-a))/200
     dz = (a - lables[n])
     dw1 += (dataset[n,0] * dz)
     dw2 += (dataset[n,1] * dz)
     db += dz
    dw1 = dw1/200
    dw2 = dw2/200
    db = db/200
    w1 += q * dw1
    w2 += q * dw2
    b += q * db
        
#print(w1,w2,b)

new = [-(w2*dataset[i,1]+ b)/w1 for i in range(200)]

plt.scatter(x = dataset[:,0],y=dataset[:,1],c=lables)
plt.plot(new,dataset[:,1])
plt.show()
