import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


boston = datasets.load_boston()
X=boston.data.astype(np.float32)
y=boston.target.astype(np.float32)
y = y.reshape(-1,1)
X = preprocessing.StandardScaler().fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)



X_p = tf.placeholder(dtype= tf.float32, shape=[None,13])
y_p = tf.placeholder(dtype= tf.float32, shape=[None,1])


Weights = tf.Variable(tf.zeros(13), dtype=tf.float32)
bias = tf.Variable(tf.zeros(1), dtype=tf.float32)

model = X_p * Weights + bias
loss_func = tf.reduce_mean((y_p - model) ** 2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss_func)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    loss_values = []
    for turn in range(5000):
        loss_value,_ = sess.run([loss_func,optimizer], feed_dict={X_p: X_train, y_p:y_train})
        loss_values.append(loss_value) 
    W_s,bias_value = sess.run([Weights,bias])


#new = W * X + bias_value
#plt.figure(0)
#plt.scatter(X,y,marker='o')
#plt.plot(X,new)
plt.figure(1)
plt.plot(loss_values)
plt.show()