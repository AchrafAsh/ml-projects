#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w_1, w_2 = 1.0, 1.0
b = 0

learning_rate = 0.01

def forward(x):
	return w_2 * x * x + w_1 * x + b


def loss(x, y):
	y_pred = forward(x)
	return (y_pred - y)*(y_pred - y)


def gradient(x, y):
	'''
	d_loss/d_w which is a vector
	'''
	y_pred = forward(x)
	return [2*x*(y_pred-y), 2*x*x*(y_pred-y)]


for epoch in range(100):
	for x_val, y_val in zip(x_data, y_data):
		grad = gradient(x_val, y_val)
		[w_1, w_2] = [w_1-learning_rate*grad[0] ,w_2-learning_rate*grad[1]]
		print("\tgrad: ", x_val, y_val, grad)
		l = loss(x_val, y_val)
	print("progress:", epoch, "w=","[", w_1, ",", w_2, "loss", l)
print("predict (after training)", "4 hours", forward(4))
print("w=", w_1, w_2)
