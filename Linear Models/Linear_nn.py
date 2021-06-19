from sklearn.datasets import make_regression
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# x_values = [i for i in range(11)]
# X = np.array(x_values, dtype=np.float32)
# X = X.reshape(-1,1)
#
# y_values = [2*i +1 for i in x_values]
# y = np.array(y_values, dtype=np.float32)
# y = y.reshape(-1,1)

X, y, coef = make_regression(n_samples=1000,
                             n_features=1,
                             noise=10,
                             coef=True)

print(X)
y = np.expand_dims(y, axis=1)
print(y)




# class LinearRegression(torch.nn.Module):
#     def __init__(self, inputSize, outputSize):
#         super(LinearRegression, self).__init__()
#         self.linear = torch.nn.Linear(inputSize, outputSize)
#
#     def forward(self, x):
#         out = self.linear(x)
#         return out
#
#
# input_dim = 1
# output_dim = 1
# lr = 0.01
# epochs = 100
#
# model = LinearRegression(input_dim, output_dim)
#
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#
# for epoch in range(epochs):
#     inputs = Variable(torch.from_numpy(X))
#     labels = Variable(torch.from_numpy(y))
#
#     optimizer.zero_grad()
#     outputs = model(inputs)
#
#     loss = criterion(outputs, labels)
#     print(loss)
#
#     loss.backward()
#
#     optimizer.step()
#
#     print(f'epoch {epoch}, loss {loss.item()}')
#
#
# with torch.no_grad():
#     predicted = model(Variable(torch.from_numpy(X))).data.numpy()
#     print(predicted)
#
# plt.clf()
# plt.plot(X, y, 'go', label='True Data', alpha=.5)
# plt.plot(X, predicted,  '--', label='Predictions', alpha=.5)
# plt.show()