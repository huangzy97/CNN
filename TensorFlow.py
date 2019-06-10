# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 21:48:52 2019
@author: sbtithzy
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# =============================================================================
# P = tf.Variable(8,name = 'param')
# M = tf.Variable(3)
# H = tf.placeholder(tf.float32)#.astype(np.float32)
# K = tf.placeholder(tf.float32)
# out = tf.multiply(H,K)
# init = tf.initialize_all_variables()
# #print(P.name)
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(out,feed_dict={H:2,K:3.}))
# =============================================================================
# add layer
def add_layer(inputs,in_size,out_size,activation_funcation = None):
    Weight = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs,Weight) + biases
    if activation_funcation is None:
        out_puts = Wx_plus_b
    else:
        out_puts = activation_funcation(Wx_plus_b)
    return out_puts
X_data = np.linspace(-1,1,500)[:,np.newaxis] 
noise = np.random.normal(0,0.05,X_data.shape)   
Y_data = np.square(X_data) -2 + noise
# 
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])


# later
layer1 = add_layer(x,1,20,activation_funcation=tf.nn.relu)

layer2 = add_layer(layer1,20,1,activation_funcation = None)
# 损失函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - layer2),reduction_indices=[1]))
# training
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 初始化所有变量 
init = tf.initialize_all_variables()
# def Session
sess = tf.Session()
sess.run(init)
# 
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(X_data,Y_data)
plt.ion()#保留图片
plt.show()
for  i in range(1000):
    sess.run(train,feed_dict={x:X_data,y:Y_data})
    if i % 10 ==0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        predict_value = sess.run(layer2,feed_dict={x:X_data})
        lines = ax.plot(X_data,predict_value,'r-',lw = 5)
        plt.pause(0.1)
        print(i,sess.run(loss,feed_dict={x:X_data,y:Y_data}))
plt.savefig(r'D:\dm\tf\pict.png')
