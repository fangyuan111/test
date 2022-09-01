import torch
a = torch.ones(2,requires_grad=True)
b = a*2
print(a, a.grad, a.requires_grad )
b.sum().backward(retain_graph = True )
print(a, a.grad, a.requires_grad )
a =a+ a.grad
print(a, a.grad, a.requires_grad )
# with torch.no_grad():
#     a += a.grad
#     print(a, a.grad, a.requires_grad )
#     a.grad.zero_()
# b.sum().backward(retain_graph = True )
# print(a, a.grad ,a.requires_grad )