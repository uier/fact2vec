import torch
from training import FactEmbedder
import sys

sys.modules['FactEmbedder'] = FactEmbedder

# net = torch.load('model/fact2vec.pth')
net = torch.load('./net_2.pth')

fact = "From February, precipitation start to decrease until July."
embedding = net(fact)
print(embedding)
# output is like:
# tensor([[-0.0560, -0.1217, -0.1291,  ...,  0.1356,  0.1849,  0.1304],
#         [ 0.0049, -0.1576, -0.0929,  ...,  0.1506,  0.1271,  0.2544],
#         [-0.0584, -0.2038, -0.1458,  ...,  0.0425,  0.1069,  0.1858],
#         ...,
#         [-0.0740, -0.1886, -0.0581,  ...,  0.1761,  0.1768,  0.2322],
#         [-0.1011, -0.2289, -0.1819,  ...,  0.1299,  0.1074,  0.1826],
#         [-0.1514, -0.2008, -0.1349,  ...,  0.1033,  0.0806,  0.1868]],
#        grad_fn=<AddmmBackward>)