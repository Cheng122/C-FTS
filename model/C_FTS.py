import torch
import torch.nn as nn
import numpy as np
import copy
import torch_dct as dct
from einops import rearrange
from collections import OrderedDict
from torch.nn import functional as F
from torch.nn import init
import scipy.sparse as sp

from timm.models.layers import DropPath


class Model(nn.Module):
    def __init__(self, args):
    # def __init__(self, layers, d_hid, frames, num_coeff_Time_kept, model_downsample_rate):
        super().__init__()
        '''
        layers: the number of stacked FTSC blocls
        d_hid: the depth of mapped channle
        num_coeff_Time_kept: the number of kept DCT coefficient
        m_ds_r: the model downsample rate
        '''

        layers, d_hid, frames = args.layers, args.d_hid, args.frames
        num_coeff_Time_kept = args.num_coeff_Time_kept                
        num_joints_in, num_joints_out = args.n_joints, args.out_joints
        m_ds_r = args.model_downsample_rate

        # layers, d_hid, frames = layers, d_hid, frames
        # num_joints_in, num_joints_out = 17, 17
        # m_ds_r = model_downsample_rate
        # num_coeff_Time_kept = num_coeff_Time_kept

        self.pose_emb = nn.Linear(2, d_hid, bias=False)                             # pose embedding：input dimension = 2 -> (x,y)，output dimension = d_hid
        self.gelu = nn.GELU()
        self.c_fts = C_FTS(layers, frames, num_joints_in, d_hid, num_coeff_Time_kept, m_ds_r)
        self.regress_head = nn.Linear(d_hid, 3, bias=False)                         # regression head: input dimension=d_hid, output dimension=3 -> (x,y,z)
        self.num_Time_kept = num_coeff_Time_kept

        self.regress_head_pre = nn.Linear((frames+m_ds_r-1)//m_ds_r + num_coeff_Time_kept, frames, bias=False) 

        self.model_ds_rate = m_ds_r


    def forward(self, x):
        # b, t, s, c = x.shape                                                      # batch,frame,joint,2

        x_t = dct.dct(x.permute(0, 2, 3, 1))[:, :, :, :self.num_Time_kept]          # DCT to Time domain, get the first "num_Time_kept" coefficients
        x_t = x_t.permute(0, 3, 1, 2).contiguous()                                  # b, n_t, s, c  

        x = x[:, ::self.model_ds_rate, :, :]                                        # b, t_d, s, c

        x = torch.cat((x, x_t), dim=1)                                              # b,t_d+n_t,s, c
        

        x = self.pose_emb(x)                                                        # b, t_d+n_t, s, d_hid
        x = self.gelu(x)

        # frequency-time-space correlation
        x = self.c_fts(x)
        # regression head
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.regress_head_pre(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.regress_head(x)

        return x


class FTSC_ATTENTION(nn.Module):
    def __init__(self, d_time, d_joint, d_coor, num_T_k, m_ds_r, head=8):
        super().__init__()
        # print(d_time, d_joint, d_coor,head)
        self.qkv = nn.Linear(d_coor, d_coor * 3)
        self.head = head
        self.layer_norm = nn.LayerNorm(d_coor)

        self.scale = (d_coor // 2) ** -0.5
        self.proj = nn.Linear(d_coor, d_coor)
        self.d_time = d_time
        self.d_joint = d_joint
        self.head = head
        self.num_T_k = num_T_k
        self.m_ds_r = m_ds_r
        self.after_downsample_frames = (d_time+m_ds_r-1)//m_ds_r

        # sep1
        # print(d_coor)
        self.emb = nn.Embedding(5, d_coor//head//2)
        self.part = torch.tensor([0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 4, 4, 4]).long().cuda()
        # self.part = torch.tensor([0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 4, 4, 4]).long()

        # sep2
        self.sep2_t = nn.Conv2d(d_coor // 2, d_coor // 2, kernel_size=3, stride=1, padding=1, groups=d_coor // 2)
        self.sep2_s = nn.Conv2d(d_coor // 2, d_coor // 2, kernel_size=3, stride=1, padding=1, groups=d_coor // 2)

        self.drop = DropPath(0.5)

    def forward(self, input):
        b, t, s, c = input.shape                                                    # batch，f_d+n_t，joints，channel

        h = input
        x = self.layer_norm(input)                                                  # batch，frame，joints，channel(norm)

        qkv = self.qkv(x)  # b, t, s, c-> b, t, s, 3*c                              # batch，frame，joints，channel*3
        qkv = qkv.reshape(b, t, s, c, 3).permute(4, 0, 1, 2, 3)  # 3,b,t,s,c        # 3, batch，frame，joints，channel

        # space group and frequency-time group
        qkv_s, qkv_t = qkv.chunk(2, 4)                                              # [3,b,t,s,c//2],  [3,b,t,s,c//2]

        q_s, k_s, v_s = qkv_s[0], qkv_s[1], qkv_s[2]                                # b,t,s,c//2
        q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2]                                # b,t,s,c//2

        # reshape for mat
        q_s = rearrange(q_s, 'b t s (h c) -> (b h t) s c', h=self.head)             # b,t,s,c//2-> b*h*t,s,c//2//h
        k_s = rearrange(k_s, 'b t s (h c) -> (b h t) c s ', h=self.head)            # b,t,s,c//2-> b*h*t,c//2//h,s

        q_t = rearrange(q_t, 'b  t s (h c) -> (b h s) t c', h=self.head)            # b,t,s,c//2 -> b*h*s,t,c//2//h
        k_t = rearrange(k_t, 'b  t s (h c) -> (b h s) c t ', h=self.head)           # b,t,s,c//2->  b*h*s,c//2//h,t

        att_s = (q_s @ k_s) * self.scale  # b*h*t,s,s                               # batch*8*frame, joints, joints
        att_t = (q_t @ k_t) * self.scale  # b*h*s,t,t                               # batch*8*joints, frame, frame

        att_s = att_s.softmax(-1)  # b*h*t,s,s
        att_t = att_t.softmax(-1)  # b*h*s,t,t

        v_s = rearrange(v_s, 'b  t s c -> b c t s ')                                # b,t,s,c//2 --> batch, c//2, fd+nt, joints
        v_t = rearrange(v_t, 'b  t s c -> b c t s ')

        # sep2 dynamic part: frequency-time-space conv
        sep2_s_DownsampleFrame = self.sep2_s(v_s[:, :, :self.after_downsample_frames, :])
        sep2_s_FreqData = self.sep2_s(v_s[:, :, self.after_downsample_frames:, :])
        sep2_s = torch.cat((sep2_s_DownsampleFrame, sep2_s_FreqData), dim=2)

        sep2_t_DownsampleFrame = self.sep2_t(v_t[:, :, :self.after_downsample_frames, :])
        sep2_t_FreqData = self.sep2_t(v_t[:, :, self.after_downsample_frames:, :])
        sep2_t = torch.cat((sep2_t_DownsampleFrame, sep2_t_FreqData), dim=2)

        sep2_s = rearrange(sep2_s, 'b (h c) t s  -> (b h t) s c ', h=self.head)     # b*h*t,s,c//2//h
        sep2_t = rearrange(sep2_t, 'b (h c) t s  -> (b h s) t c ', h=self.head)     # b*h*s,t,c//2//h

        # sep1 static part:
        sep_s = self.emb(self.part).unsqueeze(0)
        sep_t = self.emb(self.part).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # MSA
        v_s = rearrange(v_s, 'b (h c) t s  -> (b h t) s c ', h=self.head)           # b*h*t,s,c//2//h
        v_t = rearrange(v_t, 'b (h c) t s  -> (b h s) t c ', h=self.head)           # b*h*s,t,c//2//h

        x_s = att_s @ v_s + sep2_s + 0.0001 * self.drop(sep_s)[0]                   # drop(sep_s) will make [1,s,x]->[1,1,s,x]
        x_t = att_t @ v_t + sep2_t

        x_s = rearrange(x_s, '(b h t) s c -> b h t s c ', h=self.head, t=t)  # b*h*t,s,c//h//2 -> b,h,t,s,c//h//2 
        x_t = rearrange(x_t, '(b h s) t c -> b h t s c ', h=self.head, s=s)  # b*h*s,t,c//h//2 -> b,h,t,s,c//h//2 

        x_t = x_t + 1e-9 * self.drop(sep_t)

        x = torch.cat((x_s, x_t), -1)  # b,h,t,s,c//h
        x = rearrange(x, 'b h t s c -> b  t s (h c) ')  # b,t,s,c

        # projection and skip-connection
        x = self.proj(x)
        x = x + h
        return x


class FTSC_BLOCK(nn.Module):
    def __init__(self, d_time, d_joint, d_coor, num_T_k, m_ds_r):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_coor)
        self.mlp = Mlp(d_coor, d_coor * 4, d_coor)
        self.freqmlp = FreqMlp(d_coor, d_coor * 4, d_coor)

        self.ftsc_att = FTSC_ATTENTION(d_time, d_joint, d_coor, num_T_k, m_ds_r)
        self.drop = DropPath(0.0)

        self.after_downsample_frames = (d_time+m_ds_r-1)//m_ds_r

    def forward(self, input):
        b, t, s, c = input.shape                                                                # batch，fd+nt，num-joints，coordinate
        x = self.ftsc_att(input)                                                                # batch，fd+nt，num-joints，coordinate
        x1 = x[:, :self.after_downsample_frames] + self.drop(self.mlp(self.layer_norm(x[:, :self.after_downsample_frames])))
        x2 = x[:, self.after_downsample_frames:] + self.drop(self.freqmlp(self.layer_norm(x[:, self.after_downsample_frames:])))
        x = torch.cat((x1, x2), dim=1)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FreqMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        b, t, s, c = x.shape
        x = dct.idct(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = dct.dct(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        return x

class C_FTS(nn.Module):
    def __init__(self, num_block, d_time, d_joint, d_coor, num_Time_kept, m_ds_r):
        super(C_FTS, self).__init__()

        self.num_block = num_block                                                              # the num of FTSC blocks
        self.d_time = d_time                                                                    # frames
        self.d_joint = d_joint                                                                  # 2d joints = 17
        self.d_coor = d_coor                                                                    # 2
        self.num_T_k = num_Time_kept                                                            # The number of DCT coefficients retained
        self.m_ds_r = m_ds_r                                                                    # model downsample rate

        self.ftsc_block = []
        for l in range(self.num_block):
            self.ftsc_block.append(FTSC_BLOCK(self.d_time, self.d_joint, self.d_coor, self.num_T_k, self.m_ds_r))  
        self.ftsc_block = nn.ModuleList(self.ftsc_block)

    def forward(self, input):
        # blocks layers
        for i in range(self.num_block):
            input = self.ftsc_block[i](input)
        # exit()
        return input


if __name__ == "__main__":
    net = Model(layers=6, d_hid=256, frames=27, num_coeff_Time_kept=9, model_downsample_rate=3)
    inputs = torch.rand([1, 27, 17, 2])
    output = net(inputs)
    print(output.size())
    from thop import profile
    # flops = 2*macs
    macs, params = profile(net, inputs=(inputs,))
    print(2*macs)
    print(params)
