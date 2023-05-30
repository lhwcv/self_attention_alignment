# -*- coding: utf-8 -*-
# 2022 lhwcv
'''
Deep model with built-in self-attention alignment for acoustic echo cancellation

https://arxiv.org/pdf/2208.11308.pdf
'''

import torch
import torch.nn as nn


class Attention_Align_Block(nn.Module):
   def __init__(self, channels, fdim, pdim=64, max_delay_blocks=32):
       super(Attention_Align_Block, self).__init__()
       self.p_dim = pdim
       self.max_delay_blocks  = max_delay_blocks
       self.max_pooling = nn.MaxPool2d(kernel_size=(4, 1))
       in_dim = channels * fdim // 4
       self.q_proj = nn.Linear(in_dim, pdim)
       self.k_proj = nn.Linear(in_dim, pdim)

       self.neg_inf = -10e12


   def forward(self, xm, xf):
       '''

       :param xm:  B, C, F, T
       :param xf:  B, C, F, T
       :return: result: (B, C, F, T),  corr: (B, T, max_delay_blocks)
       '''
       xm_ = self.max_pooling(xm)
       xf_ = self.max_pooling(xf)
       b, c, f_div_4, t = xm_.shape

       # -> b, c*f_div_4, t -> b, t, c*f_div_4
       xm_ = xm_.reshape(b, c * f_div_4, t).permute(0, 2, 1)
       xf_ = xf_.reshape(b, c * f_div_4, t).permute(0, 2, 1)

       # -> b, t, p
       q = self.q_proj(xm_)
       k = self.k_proj(xf_)

       padd = torch.zeros((b, self.max_delay_blocks - 1, self.p_dim), dtype=k.dtype, device=k.device)
       k = torch.cat((padd, k), dim=1)

       # b, t, p --> b, t,  p, self.max_delay_blocks
       k = k.unfold(1, self.max_delay_blocks, step=1)
       # b, t, p --> b, t, p, 1
       q = q.unsqueeze(-1)
       # sum in p dim --> b, t, max_delay_blocks
       corr = (q * k).sum(2)

       # mask for previous padd
       min_t = min(corr.size(1), self.max_delay_blocks)
       mask = torch.ones(min_t, self.max_delay_blocks, device=k.device)
       mask = torch.tril(mask).flip(dims=[-1])
       mask = mask.unsqueeze(0).repeat(corr.size(0), 1, 1)
       mask = torch.logical_not(mask)
       corr[:, :self.max_delay_blocks][mask] = self.neg_inf
       corr = torch.softmax(corr, dim=-1)

       # weighted sum on Xf
       b, c, f, t = xf.shape
       padd = torch.zeros((b, c, f, self.max_delay_blocks - 1), dtype=k.dtype, device=k.device)
       xf = torch.cat((padd, xf), dim=-1)
       xf = xf.unfold(-1, self.max_delay_blocks, step=1)

       # b, t, max_delay_blocks -> b, 1, 1, t, max_delay_blocks
       corr_ = corr.unsqueeze(1).unsqueeze(1)

       out = (corr_ * xf).sum(-1)

       return  out, corr

def toy_train():
    import matplotlib.pyplot as plt

    C = 1
    F = 64
    T = 32
    shift = 2
    ref = torch.randn(1, C, F, T)
    echo = torch.roll(ref, shifts=shift, dims=-1)
    echo[:, :, :, :shift] = 0
    mic = echo

    layer = Attention_Align_Block(channels=C, fdim=F, pdim=64, max_delay_blocks=8)

    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    for i in range(1000):
        pred,_ = layer(mic, ref)
        loss = loss_fn(pred, echo)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i %100 ==0:
            print('loss: ', loss.item())

    with torch.no_grad():
        out, corr = layer(mic, ref)
        plt.imshow(corr[0].detach().cpu().numpy().T)
        plt.show()

if __name__ == '__main__':
    toy_train()
