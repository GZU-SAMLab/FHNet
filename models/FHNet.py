import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models
import numpy as np
from .backbones import ResNet, Conv_4
from .module.SQIL import SQIL
from .module.util import Attention_E
from .module.DFIE_FSHL import FSHL, DFIE, Transformer

class FHNet(nn.Module):
    
    def __init__(self,way=None,shots=None,resnet=False,args=None):
        
        super().__init__()

        self.resolution = 5*5
        self.args = args
        if args.dataset == 'cub':
            self.classes = 200
        elif args.dataset == 'cars':
            self.classes = 196
        elif args.dataset == 'dogs':
            self.classes = 120
        elif args.dataset == "FD-PDC":
            self.classes = 51

        if resnet:
            self.num_channel = 640
            self.feature_extractor = ResNet.resnet12()
            self.dim = self.num_channel*5*5
            
        else:
            self.num_channel = 64
            self.feature_extractor = Conv_4.BackBone(self.num_channel)            
            self.dim = self.num_channel*5*5


        self.shots = shots
        self.way = way
        self.resnet = resnet

        self.scale_s = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)
        self.scale_f = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)
        self.w1 = nn.Parameter(torch.FloatTensor([0.5]),requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor([0.5]),requires_grad=True)
        self.fmrm = SQIL(hidden_size=self.num_channel, inner_size=self.num_channel, num_patch=self.resolution, drop_prob=0.1)
        self.attention = Attention_E(embedding_dim=self.num_channel, drop_prob=0.1)
        self.fshl = FSHL()
    
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.clasifier1 = nn.Linear(self.num_channel, self.classes) 
        self.clasifier2 = nn.Linear(self.num_channel, self.classes)
        
        

    def get_feature_vector(self,inp):
        freq_feature, rgb_feature = self.fshl(inp.cpu())
        rgb_feature = self.attention(rgb_feature)

        return rgb_feature, freq_feature
    

    def get_neg_l2_dist(self,inp,way,shot,query_shot):
        B, C, H, W = inp.shape
        rgb_feature, freq_feature = self.get_feature_vector(inp)  

        support_s = rgb_feature[:way*shot].view(way, shot, *rgb_feature.size()[1:]).permute(0, 2, 1, 3, 4).contiguous()
        query_s = rgb_feature[way*shot:]

        support_f = freq_feature[:way*shot].view(way, shot, *freq_feature.size()[1:]).permute(0, 2, 1, 3, 4).contiguous()
        query_f = freq_feature[way*shot:]

        sq_similarity_s, qs_similarity_s = self.fmrm(support_s, query_s)
        l2_dist_s = self.w1*sq_similarity_s + self.w2*qs_similarity_s

        sq_similarity_f, qs_similarity_f = self.fmrm(support_f, query_f)
        l2_dist_f = self.w1*sq_similarity_f + self.w2*qs_similarity_f
        
        
        return l2_dist_s, l2_dist_f

    
    def meta_test(self,inp,way,shot,query_shot):

        neg_l2_dist_s, neg_l2_dist_f = self.get_neg_l2_dist(inp=inp,
                                        way=way,
                                        shot=shot,
                                        query_shot=query_shot)
        neg_l2_dist_all = neg_l2_dist_s + neg_l2_dist_f
        _,max_index = torch.max(neg_l2_dist_all, 1)
        return max_index


    def forward(self,inp):
        logits_s, logits_f = self.get_neg_l2_dist(inp=inp,
                                        way=self.way,
                                        shot=self.shots[0],
                                        query_shot=self.shots[1])
        logits_s = logits_s/self.dim*self.scale_s
        logits_f = logits_f/self.dim*self.scale_f
        log_prediction_s = F.log_softmax(logits_s,dim=1)
        log_prediction_f = F.log_softmax(logits_f,dim=1)
        return log_prediction_s, log_prediction_f
