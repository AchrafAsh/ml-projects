#coding=utf-8
#mnist.py

import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# MNIST Dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081,))]), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

# Data loader (Input pipeline)
batch_size=10
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

class SimpleNet(torch.nn.Module):
	def __init__(self):
		super(SimpleNet, self).__init__()
		self.l1 = torch.nn.Linear(28*28, 520)
		self.l2 = torch.nn.Linear(520, 320)
		self.l3 = torch.nn.Linear(320, 10)

	def forward(self, x):
		# Flatten the data (n, 1, 28, 28) => (n, 784)
		x = x.view(-1, 28*28)
		output = F.relu(self.l1(x))
		output = F.relu(self.l2(output))
		return self.l3(output) # no need of activation function as cross entropy has it built-in


class ConvNet(torch.nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
		self.mp = torch.nn.MaxPool2d(2)
		self.fc = torch.nn.Linear(320, 10) # fully connected layer

	def forward(self, x):
		in_size = x.size(0)
		x = F.relu(self.mp(self.conv1(x)))
		x = F.relu(self.mp(self.conv2(x)))
		x = x.view(in_size, -1) # flatten the tensor
		x = self.fc(x)
		return F.log_softmax(x)


# model = SimpleNet()
model = ConvNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % 10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.item()))

def test():
	model.eval()
	test_loss = 0
	correct = 0
	for data, target in test_loader:
		data, target = Variable(data), Variable(target)
		output = model(data)

		test_loss += criterion(output, target).item()
		pred = torch.max(output.data, 1)[1]
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()

	test_loss /= len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100.*correct/len(test_loader.dataset)))


for epoch in range(1, 10):
	train(epoch)
	test()
