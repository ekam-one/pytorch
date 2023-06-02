import torch
import torch.nn.functional as F


def fn(x, y):
    # return torch.sin(x)
    return F.softmax(x, dim=1)
    # return torch.sum(x)


input_tensor = torch.randn(2, 128, requires_grad=True)
print("calling compile****************************")
# breakpoint()
model_opt = torch.compile(fn, backend="torchmhlo")
# model_opt = torch.compile(fn)
print("calling compiled_ref****************************")
# breakpoint()
out = model_opt(input_tensor, input_tensor)
print("calling compiled_ref over****************************")
print(out)

print(out.sum().backward())
