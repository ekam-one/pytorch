import torch


def fn(x, y):
    return x / y


input_tensor = torch.range(0, 99, requires_grad=True)
model_opt = torch.compile(fn, backend="torchmhlo")
out = model_opt(input_tensor, input_tensor)
print(out)

print(out.sum().backward())
