#coding=utf-8
import torch
from torch.autograd import Variable

class LinearNet(torch.nn.Module):
	def __init__(self):
		super(LinearNet, self).__init__()
		self.linear = torch.nn.Linear(1, 1) # dimensions: 1 in | 1 out

	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

model = LinearNet()

# loss function
criterion = torch.nn.MSELoss(size_average=False)
# define how the model learns, lr=learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training loop
for epoch in range(500):
	# forward pass: compute predicted y by passing x to the model
	y_pred = model(x_data)

	loss = criterion(y_pred, y_data)
	print(epoch, loss.item())

	# zero gradients, perform a backward pass and update the weights
	optimizer.zero_grad()
	loss.backward() # compute gradients
	optimizer.step() # update parameters

hour_var = Variable(torch.Tensor([[4.0]]))
print("predict (after training)", 4, model.forward(hour_var).data[0][0])
