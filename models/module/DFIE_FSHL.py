import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backbones.ResNet import resnet12
from backbones.Conv_4 import BackBone as Conv4
from .utils import PreNorm, FeedForward, Attention, Share_Attention
from einops import rearrange
import torch_dct as DCT
import cv2
import numpy as np

def norm(x):
    return (1 - torch.exp(-x)) / (1 + torch.exp(-x))

def norm_(x):
    import numpy as np
    return (1 - np.exp(-x)) / (1 + np.exp(-x))

class ImageProcessor:
    def __init__(self, block_size):
        self.block_size = block_size

    def pad_and_split_blocks(self, tensor, block_size):
        # [batch, channels, height, width]
        batch, channels, height, width = tensor.shape
        pad_height = (block_size - (height % block_size)) % block_size
        pad_width = (block_size - (width % block_size)) % block_size

        # padding
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_width, 0, pad_height))
        padded_height, padded_width = padded_tensor.shape[2], padded_tensor.shape[3]

        # patch
        blocks = padded_tensor.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        blocks = blocks.permute(0, 2, 3, 1, 4, 5).contiguous()
        return blocks

    def rgb_to_ycbcr(self, x):
        x_np = x.numpy()
        x_np = np.transpose(x_np, (0, 2, 3, 1))  

        ycbcr_images = np.zeros_like(x_np)
        for i in range(x_np.shape[0]):
            ycbcr_images[i] = cv2.cvtColor(x_np[i], cv2.COLOR_RGB2YCrCb)

        ycbcr_images = np.transpose(ycbcr_images, (0, 3, 1, 2)) 
        x_ycbcr = torch.from_numpy(ycbcr_images).float() 

        num_batchsize, _, size, _ = x_ycbcr.shape

        x_ycbcr = self.pad_and_split_blocks(x_ycbcr, self.block_size)
        x_ycbcr = DCT.dct_2d(x_ycbcr, norm='ortho')
        x_ycbcr = torch.nn.functional.normalize(x_ycbcr, p=2, dim=3) 
        x_ycbcr = x_ycbcr.reshape(num_batchsize, size // 8 + 1, size // 8 + 1, -1).permute(0, 3, 1, 2)
        return x_ycbcr

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class channel_shuffle(nn.Module):
    def __init__(self,groups=4):
        super(channel_shuffle,self).__init__()
        self.groups=groups
    def forward(self,x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups

        x = x.view(batchsize, self.groups,
               channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()

        x = x.view(batchsize, -1, height, width)
        return x

class two_ConvBnRule(nn.Module):

    def __init__(self, in_chan, out_chan=64):
        super(two_ConvBnRule, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN1 = nn.BatchNorm2d(out_chan)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN2 = nn.BatchNorm2d(out_chan)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, mid=False):
        feat = self.conv1(x)
        feat = self.BN1(feat)
        feat = self.relu1(feat)

        if mid:
            feat_mid = feat

        feat = self.conv2(feat)
        feat = self.BN2(feat)
        feat = self.relu2(feat)

        if mid:
            return feat, feat_mid
        return feat

def Seg():
    dict = {0: 0, 1: 1, 2: 8, 3: 16, 4: 9, 5: 2, 6: 3, 7: 10, 8: 17,
                 9: 24, 10: 32, 11: 25, 12: 18, 13: 11, 14: 4, 15: 5, 16: 12,
                 17: 19, 18: 26, 19: 33, 20: 40, 21: 48, 22: 41, 23: 34, 24: 27,
                 25: 20, 26: 13, 27: 6, 28: 7, 29: 14, 30: 21, 31: 28, 32: 35,
                 33: 42, 34: 49, 35: 56, 36: 57, 37: 50, 38: 43, 39: 36, 40: 29,
                 41: 22, 42: 15, 43: 23, 44: 30, 45: 37, 46: 44, 47: 51, 48: 58,
                 49: 59, 50: 52, 51: 45, 52: 38, 53: 31, 54: 39, 55: 46, 56: 53,
                 57: 60, 58: 61, 59: 54, 60: 47, 61: 55, 62: 62, 63: 63}
    a = torch.zeros(1, 64, 1, 1)

    for i in range(0, 32):
        a[0, dict[i+32], 0, 0] = 1
    return a


class FFS(nn.Module):

    def __init__(self, in_dim):

        super(FFS, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_dim*2,
            out_channels= 2,
            kernel_size=3,
            padding=1
        )

        self.v_rgb = nn.Parameter(torch.randn((1,in_dim,1,1)),requires_grad=True)
        self.v_freq = nn.Parameter(torch.randn((1,in_dim,1,1)),requires_grad=True)

    def forward(self, rgb, freq):

        attmap = self.conv( torch.cat( (rgb,freq),1) )
        attmap = torch.sigmoid(attmap)

        rgb = attmap[:,0:1,:,:] * rgb * self.v_rgb
        freq = attmap[:,1:,:,:] * freq * self.v_freq
        out = rgb + freq

        return out

class DFIE(nn.Module):
    def __init__(self):
        super(DFIE, self).__init__()
        self.processor = ImageProcessor(block_size=8)
        self.seg = Seg().cuda()  
        self.share_attention = Share_Attention(dim=256, heads=4, dim_head=128, dropout=0., alpha=0.5).cuda()
        self.high_band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128*2, dropout=0).cuda()
        self.low_band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128*2, dropout=0).cuda()
        self.band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128*2, dropout=0).cuda()
        self.spatial = Transformer(dim=192, depth=1, heads=2, dim_head=64, mlp_dim=64*2, dropout=0).cuda()

        self.shuffle=channel_shuffle()

        self.vector_y = nn.Parameter(torch.FloatTensor(1, 64, 1, 1), requires_grad=True)
        self.vector_cb = nn.Parameter(torch.FloatTensor(1, 64, 1, 1), requires_grad=True)
        self.vector_cr = nn.Parameter(torch.FloatTensor(1, 64, 1, 1), requires_grad=True)

    def forward(self, x):
        # if x.is_cuda is False: 
        #     x = x.cuda()
        
        DCT_x = self.processor.rgb_to_ycbcr(x).cuda()
        self.seg = self.seg.to(DCT_x.device)  

        feat_y = DCT_x[:, 0:64, :, :] * (self.seg + norm(self.vector_y))
        feat_Cb = DCT_x[:, 64:128, :, :] * (self.seg + norm(self.vector_cb))
        feat_Cr = DCT_x[:, 128:192, :, :] * (self.seg + norm(self.vector_cr))

        origin_feat_DCT = torch.cat((torch.cat((feat_y, feat_Cb), 1), feat_Cr), 1)
        origin_feat_DCT = self.shuffle(origin_feat_DCT).cuda()

        high = torch.cat([feat_y[:, 32:, :, :], feat_Cb[:, 32:, :, :], feat_Cr[:, 32:, :, :]], 1)
        low = torch.cat([feat_y[:, :32, :, :], feat_Cb[:, :32, :, :], feat_Cr[:, :32, :, :]], 1)

        b, n, h, w = high.shape
        high = torch.nn.functional.interpolate(high, size=(16, 16)).cuda()
        low = torch.nn.functional.interpolate(low, size=(16, 16)).cuda()

        high = rearrange(high, 'b n h w -> b n (h w)')
        low = rearrange(low, 'b n h w -> b n (h w)')

        high, low = self.share_attention(high, low)
        # high = self.high_band(high) 
        # low = self.low_band(low) 

        y_h, b_h, r_h = torch.split(high, 32, 1)
        y_l, b_l, r_l = torch.split(low, 32, 1)
        feat_y = torch.cat([y_l, y_h], 1)
        feat_Cb = torch.cat([b_l, b_h], 1)
        feat_Cr = torch.cat([r_l, r_h], 1)

        feat_DCT = torch.cat((torch.cat((feat_y, feat_Cb), 1), feat_Cr), 1)

        feat_DCT = self.band(feat_DCT)
        feat_DCT = feat_DCT.transpose(1, 2)
        feat_DCT = self.spatial(feat_DCT)
        feat_DCT = feat_DCT.transpose(1, 2)
        feat_DCT = rearrange(feat_DCT, 'b n (h w) -> b n h w', h=16)
        feat_DCT = torch.nn.functional.interpolate(feat_DCT, size=(h, w))

        feat_DCT = origin_feat_DCT + feat_DCT

        return feat_DCT

class FSHL(nn.Module):

    def __init__(self, resent=True):
        super(FSHL, self).__init__()
        if resent:
            self.backbone = resnet12()
            self.dim1 = 64
            self.dim2 = 160
            self.dim3 = 320
            self.dim4 = 640
        else:
            self.backbone = Conv4()
            self.dim1 = 64
            self.dim2 = 64
            self.dim3 = 64
            self.dim4 = 64


        self.dfie = DFIE()
 
        self.conv_l2 = two_ConvBnRule(self.dim1, self.dim1).cuda()
        self.conv_l3 = two_ConvBnRule(self.dim2, self.dim2).cuda()
        self.conv_l4 = two_ConvBnRule(self.dim3, self.dim3).cuda()
        self.conv_l5 = two_ConvBnRule(self.dim4, self.dim4).cuda()

        self.conv_decoder1 = two_ConvBnRule(128).cuda()
        self.conv_decoder2 = two_ConvBnRule(128).cuda()
        self.conv_decoder3 = two_ConvBnRule(128).cuda()

        self.FFS2 = FFS(in_dim=self.dim1).cuda()
        self.FFS3 = FFS(in_dim=self.dim2).cuda()
        self.FFS4 = FFS(in_dim=self.dim3).cuda()
        self.FFS5 = FFS(in_dim=self.dim4).cuda()

        self.conv_r2 = two_ConvBnRule(self.dim1, self.dim1).cuda()
        self.conv_r3 = two_ConvBnRule(self.dim2, self.dim2).cuda()
        self.conv_r4 = two_ConvBnRule(self.dim3, self.dim3).cuda()
        self.conv_r5 = two_ConvBnRule(self.dim4, self.dim4).cuda()

        self.con1_2 = nn.Conv2d(in_channels=192, out_channels=self.dim1, kernel_size=1).cuda()
        self.con1_3 = nn.Conv2d(in_channels=192, out_channels=self.dim2, kernel_size=1).cuda()
        self.con1_4 = nn.Conv2d(in_channels=192, out_channels=self.dim3, kernel_size=1).cuda()
        self.con1_5 = nn.Conv2d(in_channels=192, out_channels=self.dim4, kernel_size=1).cuda()

        self.conv_2 = nn.Sequential(
            nn.Conv2d(self.dim1, self.dim2, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.dim2, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2)
        ).cuda()
        self.conv_3 = nn.Sequential(
            nn.Conv2d(self.dim2, self.dim3, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.dim3, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2)
        ).cuda()
        self.conv_4 = nn.Sequential(
            nn.Conv2d(self.dim3, self.dim4, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.dim4, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2)
        ).cuda()
        self.conv_5 = nn.Sequential(
            nn.Conv2d(self.dim4, self.dim4, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.dim4, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),
        ).cuda()
        


    def forward(self, x, name=None):
        if x.device.type == 'cpu':
            x = x.cuda()
        size = x.size()[2:]
        feat1 = x
        feat2, feat3, feat4, feat5 = self.backbone(feat1)

        rgb_feat_2 = feat2.cuda()
        rgb_feat_3 = feat3.cuda()
        rgb_feat_4 = feat4.cuda()
        rgb_feat_5 = feat5.cuda()

        # Module_s
        feat2 = self.conv_l2(feat2.cuda())
        feat3 = self.conv_l3(feat3.cuda())
        feat4 = self.conv_l4(feat4.cuda())
        feat5 = self.conv_l5(feat5.cuda())
        
        feat_DCT = self.dfie(x.cpu())
        if feat_DCT.device.type == 'cpu':
            feat_DCT = feat_DCT.cuda()

        feat_DCT2 = self.con1_2(feat_DCT)
        feat_DCT3 = self.con1_3(feat_DCT)
        feat_DCT4 = self.con1_4(feat_DCT)
        feat_DCT5 = self.con1_5(feat_DCT)

        feat_DCT2 = torch.nn.functional.interpolate(feat_DCT2,size=feat2.size()[2:],mode='bilinear',align_corners=True)
        feat_DCT3 = torch.nn.functional.interpolate(feat_DCT3,size=feat3.size()[2:],mode='bilinear',align_corners=True)
        feat_DCT4 = torch.nn.functional.interpolate(feat_DCT4,size=feat4.size()[2:],mode='bilinear',align_corners=True)
        feat_DCT5 = torch.nn.functional.interpolate(feat_DCT5,size=feat5.size()[2:],mode='bilinear',align_corners=True)

        feat2 = self.FFS2(feat2, feat_DCT2)
        feat3 = self.FFS3(feat3, feat_DCT3)
        feat4 = self.FFS4(feat4, feat_DCT4)
        feat5 = self.FFS5(feat5, feat_DCT5)

        feat2 = self.conv_r2(feat2)
        feat3 = self.conv_r3(feat3)
        feat4 = self.conv_r4(feat4)
        feat5 = self.conv_r5(feat5)

        feat2 = self.conv_2(feat2)
        feat3 = self.FFS3(feat2, feat3)
        # feat3 = feat2 + feat3
        
        rgb_feat_2 = self.conv_2(rgb_feat_2)
        rgb_feat_3 = self.FFS3(rgb_feat_2, rgb_feat_3)

        feat3 = self.conv_3(feat3)
        feat4 = self.FFS4(feat3, feat4)
        # feat4 = feat3 + feat4

        rgb_feat_3 = self.conv_3(rgb_feat_3)
        rgb_feat_4 = self.FFS4(rgb_feat_3, rgb_feat_4)

        feat4 = self.conv_4(feat4)
        feat5 = self.FFS5(feat4, feat5)
        # feat5 = feat4 + feat5

        rgb_feat_4 = self.conv_4(rgb_feat_4)
        rgb_feat_5 = self.FFS5(rgb_feat_4, rgb_feat_5)

        feat5 = self.conv_5(feat5)
        rgb_feat_5 = self.conv_5(rgb_feat_5)

        return feat5, rgb_feat_5

if __name__ == "__main__":
    x = torch.randn(300, 3, 84, 84)
    fshl = FSHL()
    freq, rgb = fshl(x)
    print(freq.shape)
    print(rgb.shape)