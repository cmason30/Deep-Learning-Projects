import numpy as np
import math

x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# random weights

a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

lr = 1e-6

for t in range(2000):
    # forward pass: compute predicted y
    # y = a + bx + cx^2 + dx^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # back prop to compute grads of weights
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    a -= lr * grad_a
    b -= lr * grad_b
    c -= lr * grad_c
    d -= lr * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')

