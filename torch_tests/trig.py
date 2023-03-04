import torch


def fn(x, y):
    a = torch.sin(x)
    b = torch.sin(y)
    return a + b


new_fn = torch.compile(fn, backend="inductor")
input_tensor = torch.randn(100)
print(new_fn(input_tensor, input_tensor))
