"""From ChemNet:

https://github.com/baker-laboratory/PLACER/blob/main/modules/attention.py
"""


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from opt_einsum import contract as einsum
except:
    from torch import einsum


def lecun_normal(module, scale=1.0):
    def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
        normal = torch.distributions.normal.Normal(0, 1)

        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma

        alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
        p = alpha_normal_cdf + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform

        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
        x = torch.clamp(x, a, b)

        return x

    def sample_truncated_normal(shape, scale=1.0):
        stddev = np.sqrt(scale/shape[-1])/.87962566103423978  # shape[-1] = fan_in
        return stddev * truncated_normal(torch.rand(shape))

    module.weight = torch.nn.Parameter( (sample_truncated_normal(module.weight.shape)) )
    return module


# ============================================================
class Dropout(nn.Module):
    # Dropout entire row or column
    def __init__(self, broadcast_dim=None, p_drop=0.15):
        super(Dropout, self).__init__()
        # give ones with probability of 1-p_drop / zeros with p_drop
        self.sampler = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-p_drop]))
        self.broadcast_dim=broadcast_dim
        self.p_drop=p_drop
    def forward(self, x):
        if not self.training: # no drophead during evaluation mode
            return x
        shape = list(x.shape)
        if not self.broadcast_dim == None:
            shape[self.broadcast_dim] = 1
        mask = self.sampler.sample(shape).to(x.device).view(shape)
        
        x = mask * x / (1.0 - self.p_drop)
        return x


# ============================================================
class FeedForwardLayer(nn.Module):
    def __init__(self, d_in, d_out, r_ff=4, p_drop=0.1, normalize=True):
        super(FeedForwardLayer, self).__init__()
        if normalize==True:
            self.norm = nn.LayerNorm(d_in)
        else:
            self.norm = None
        self.linear1 = nn.Linear(d_in, d_out*r_ff)
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_out*r_ff, d_out)

        self.reset_parameter()

    def reset_parameter(self):
        # initialize linear layer right before ReLu: He initializer (kaiming normal)
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)
        nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)

        # initialize linear layer right before residual connection: zero initialize
        #nn.init.zeros_(self.linear2.weight)
        #nn.init.zeros_(self.linear2.bias)

    def forward(self, src):
        if self.norm is not None:
            src = self.norm(src)
        src = self.linear2(self.dropout(F.relu_(self.linear1(src))))
        return src


# ============================================================
class BiasedAxialAttention(nn.Module):
    '''tied axial attention with bias from coordinates'''
    def __init__(self, d_pair, d_bias, n_head, d_hidden, is_row=True):
        
        super(BiasedAxialAttention, self).__init__()
        #
        self.is_row = is_row
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_bias = nn.LayerNorm(d_bias)

        self.to_q = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False) 
        self.to_g = nn.Linear(d_pair, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_pair)
        
        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        
        # initialize all parameters properly
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # bias: normal distribution
        self.to_b = lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, pair, bias, mask):
        # pair: (B, L, L, d_pair)
        B, L = pair.shape[:2]
        
        if self.is_row:
            pair = pair.permute(0,2,1,3)
            bias = bias.permute(0,2,1,3)

        pair = self.norm_pair(pair)
        bias = self.norm_bias(bias)
        
        query = self.to_q(pair).reshape(B, L, L, self.h, self.dim)
        key = self.to_k(pair).reshape(B, L, L, self.h, self.dim)
        value = self.to_v(pair).reshape(B, L, L, self.h, self.dim)
        bias = self.to_b(bias) # (B, L, L, h)
        gate = torch.sigmoid(self.to_g(pair)) # (B, L, L, h*dim) 
        
        query = query * self.scaling
        key = key / math.sqrt(L) # normalize for tied attention
        attn = einsum('bnihk,bnjhk->bijh', query, key) # tied attention
        attn = attn + bias # apply bias
        attn = attn - 1e3*(~mask[...,None]) # apply mask
        attn = F.softmax(attn, dim=-2) # (B, L, L, h)
        
        out = einsum('bijh,bkjhd->bikhd', attn, value).reshape(B, L, L, -1)
        out = gate * out
        
        out = self.to_out(out)
        if self.is_row:
            out = out.permute(0,2,1,3)

        return out

    
# ============================================================
class PairStr2Pair(nn.Module):
    def __init__(self, d_pair=128, n_head=4, d_hidden=32, d_rbf=36, p_drop=0.15):
        super(PairStr2Pair, self).__init__()

        self.emb_rbf = nn.Linear(d_rbf, d_hidden)
        self.proj_rbf = nn.Linear(d_hidden, d_pair)

        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)

        self.row_attn = BiasedAxialAttention(d_pair, d_pair, n_head, d_hidden, is_row=True)
        self.col_attn = BiasedAxialAttention(d_pair, d_pair, n_head, d_hidden, is_row=False)

        self.ff = FeedForwardLayer(d_pair, d_pair, 2)

        self.norm = nn.LayerNorm(d_pair)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.kaiming_normal_(self.emb_rbf.weight, nonlinearity='relu')
        nn.init.zeros_(self.emb_rbf.bias)

        self.proj_rbf = lecun_normal(self.proj_rbf)
        nn.init.zeros_(self.proj_rbf.bias)

    def forward(self, pair, rbf_feat, mask=None):
        if mask is None:
            mask = torch.full(pair.shape[:2], True, device=pair.device)

        # add a batch dimension to both inputs
        # such that operations on square matrices are applied to all pairs
        # and do not confuse rows as the batch dimension
        pair = pair[None]
        mask = mask[None]
        rbf_feat = rbf_feat[None]

        rbf_feat = self.proj_rbf(F.relu_(self.emb_rbf(rbf_feat)))

        pair = pair + self.drop_row(self.row_attn(pair, rbf_feat, mask))
        pair = pair + self.drop_col(self.col_attn(pair, rbf_feat, mask))
        pair = self.norm(pair + self.ff(pair)) # I.A. added LayerNorm

        pair = pair.squeeze(0)  # remove the batch dimension
        return pair