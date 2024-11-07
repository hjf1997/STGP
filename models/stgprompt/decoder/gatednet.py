import torch.nn as nn
import torch.nn.functional as F
import torch

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class GatedNet(nn.Module):
    def __init__(self, dropout=0.3,
                 gcn_bool=True,
                 in_dim=1,
                 out_dim=12,
                 residual_channels=32,
                 dilation_channels=32,
                 skip_channels=128,
                 end_channels=512,
                 kernel_size=2,
                 blocks=2,
                 layers=3,
                 supports_len=2):
        super(GatedNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool

        self.filter_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gate1_convs = nn.ModuleList()
        self.gate2_convs = nn.ModuleList()
        self.ln = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        receptive_field = 1

        # All supports are double transition
        self.supports_len = supports_len

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

                # spatial and temporal integration gate
                self.gate1_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, 1), bias=False))

                self.gate2_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, 1)))

                self.ln.append(nn.LayerNorm(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.receptive_field = receptive_field

    def forward(self, input, supports):
        # input : [B, N, L, D]
        input = input.permute(0,3,1,2)
        # input : [B, D, N, L]
        in_len = input.size(3)
        # if in_len<self.receptive_field:
        x = nn.functional.pad(input,(14,0,0,0)) # todo: change
        # else:
        x = self.start_conv(x)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            t_x = self.filter_convs[i](residual)
            s_x = self.gconv[i](residual, supports)
            s_x = s_x[: ,: ,:, -t_x.size(3):]

            # spatial and temporal integration gate
            gate = torch.sigmoid(self.gate1_convs[i](t_x) + self.gate2_convs[i](s_x))
            x = s_x * gate + t_x * (1 - gate)

            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = x + residual[:, :, :, -x.size(3):]

            x = x.permute([0, 2, 3, 1])
            x = self.ln[i](x)
            x = x.permute([0, 3, 1, 2])
        return skip
