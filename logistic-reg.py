#coding=utf-8
import torch
from torch.autograd import Variable
import torch.nn.functional as F


x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.Tensor([[0.], [0.], [1.], [1.]]))


class LogNet(torch.nn.Module):
	def __init__(self):
		super(LogNet, self).__init__()
		self.linear = torch.nn.Linear(1, 1)

	def forward(self, x):
		# sigmoid: activation function (ReLU, Tanh, ELU, SELU...)
		y_pred = F.sigmoid(self.linear(x))
		return y_pred

model = LogNet()

criterion = torch.nn.BCELoss(size_average=True) # binary cross entropy
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
	y_pred = model(x_data)
	loss = criterion(y_pred, y_data)
	print(epoch, loss.item())

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

hour_var = Variable(torch.Tensor([[1.0]]))
print("predict 1 hour ", 1.0, model(hour_var).data[0][0] > 0.5)
hour_var = Variable(torch.Tensor([[7.0]]))
print("predict 7 hour ", 7.0, model(hour_var).data[0][0] > 0.5)
