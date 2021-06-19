import torch
import math

dtype1 = torch.float
device = torch.device('cpu')

# Random data

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype1)
y = torch.sin(x)

a = torch.randn((), device=device, dtype=dtype1, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype1, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype1, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype1, requires_grad=True)

lr = 1e-6
# manual_backprop = False
for t in range(2000):
    # Forward pass
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute loss
    loss = (y_pred - y).pow(2).sum()
    if t%100 == 99:
        print(t, loss.item())

    # Backprop
    # if manual_backprop:
    #     grad_y_pred = 2.0 * (y_pred - y)
    #     grad_a = grad_y_pred.sum()
    #     grad_b = (grad_y_pred * x).sum()
    #     grad_c = (grad_y_pred * x ** 2).sum()
    #     grad_d = (grad_y_pred * x ** 3).sum()
    #
    #     a -= lr * grad_a
    #     b -= lr * grad_b
    #     c -= lr * grad_c
    #     d -= lr * grad_d
    # else:
    loss.backward()
    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad
        c -= lr * c.grad
        d -= lr * d.grad

        # Manually zero the grads after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')

