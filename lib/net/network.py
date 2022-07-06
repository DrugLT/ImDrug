import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np
import os
import utils
# from backbone import 
from neck import GAP, Identity, Concat
from head import LWS, FCNorm, MLP

class Network(nn.Module):
    """Wrapper of DeepPurpose model

    """
    def __init__(self, model, cfg, mode="train", setting_type="LT Classification", entity_type=['drug'],
                num_class=10):
        super(Network, self).__init__()
        pretrain = (
            True
            if mode == "train"
            and cfg['resume_model'] == ""
            and cfg['backbone']['pretrain'] 
            and cfg['backbone']['drug_pretrained_model'] != '' or cfg['backbone']['protein_pretrained_model'] != '' 
            else False
        )
        self.model = model
        self.num_class = num_class
        self.cfg = cfg
        self.input_kws = entity_type
        self.backbone_kws = ['model_' + input_name for input_name in entity_type]

        
        self._get_backbone()
        self.mode = mode
        self.neck = self._get_neck()
        self.head = self._get_head()
        self.networks = list(self.backbone.values()) + [self.neck] + [self.head]

        # if cfg['network']['pretrained'] and os.path.isfile(cfg['network']['pretrained_model']):
        #     try:
        #         self.load_model(cfg['network']['pretrained_model'])
        #     except:
        #         raise ValueError('network pretrained model error')

    def forward(self, x, **kwargs):
        if "feature_flag" in kwargs or "feature_cb" in kwargs or "feature_rb" in kwargs:
            return self.extract_feature(x, **kwargs)
        elif "head_flag" in kwargs:
            return self.head(x)
        elif 'feature_maps_flag' in kwargs:
            return self.extract_feature_maps(x)
        elif 'layer' in kwargs and 'index' in kwargs:
            if kwargs['layer'] in ['layer1', 'layer2', 'layer3']:
                x = self.backbone.forward(x, index=kwargs['index'], layer=kwargs['layer'], coef=kwargs['coef'])
            else:
                x = self.backbone(x)
            x = self.module(x)
            if kwargs['layer'] == 'pool':
                x = kwargs['coef']*x+(1-kwargs['coef'])*x[kwargs['index']]
            x = x.view(x.shape[0], -1)
            x = self.head(x)
            if kwargs['layer'] == 'fc':
                x = kwargs['coef']*x + (1-kwargs['coef'])*x[kwargs['index']]
            return x

        x = self.extract_feature(x)
        x = self.head(x)
        return x

    def to(self, device=None):
        if device is None:
            device = utils.device
        for net in self.networks:
            net.to(device)
            # net.apply(utils.init_weight)

    def freeze_backbone(self):
        print("Freezing backbone .......")
        for key, _ in self.backbone.items():
            for p in self.backbone[key].parameters():
                p.requires_grad = False
    
    def load_backbone_model(self, backbone_paths):
        for key, backbone_path in backbone_paths.items():
            print("Loading Backbone {} pretrain model from {}......".format(key, backbone_path))
            model_dict = self.backbone[key].state_dict()
            pretrain_dict = torch.load(backbone_path)
            pretrain_dict = pretrain_dict["state_dict"] if "state_dict" in pretrain_dict else pretrain_dict
            from collections import OrderedDict

            new_dict = OrderedDict()
            for k, v in pretrain_dict.items():
                if k.startswith("module"):
                    k = k[7:]
                if "fc" not in k and "head" not in k:
                    k = k.replace("encoder_q.", "")
                    k = k.replace("backbone.", "")
                    new_dict[k] = v

            model_dict.update(new_dict)
            self.load_state_dict(model_dict)
            print("Backbone model {} has been loaded......".format(key))
    
    def load_model(self, model_file):
        self.load_state_dict(torch.load(model_file)['state_dict'])

    def extract_feature(self, inputs:list, **kwargs):
        x = self.extract_feature_maps(inputs)
        x = x.view(x.shape[0], -1)
        return x

    def extract_feature_maps(self, inputs: list):
        feats = []
        for x, backbone_kw in zip(inputs, self.backbone_kws):
            if 'protein' in backbone_kw and self.cfg['dataset']['protein_encoding'] == 'Transformer':
                x = torch.transpose(x, 0, 1)
            elif 'drug' in backbone_kw  and self.cfg['dataset']['drug_encoding'] == 'Transformer':
                x = torch.transpose(x, 0, 1)
            feat = self.backbone[backbone_kw](x)

            feats.append(feat)
        x = self.neck(feats, self.input_kws)
        return x

    def _get_backbone(self):
        self.backbone = {}
        for backbone_kw in set(self.backbone_kws):
            assert hasattr(self.model, backbone_kw), 'no backbone model for input type {}'.format(backbone_kw)
            self.backbone[backbone_kw] = getattr(self.model, backbone_kw)

    def _get_neck(self):
        layer_type = self.cfg['neck']['type']
        if layer_type == "GAP":
            layer = GAP()
        elif layer_type == "Identity":
            layer = Identity()
        elif layer_type == "Concat":
            layer = Concat() 
        else:
            raise NotImplementedError

        return layer

    def _get_head(self):
        bias_flag = self.cfg['head']['bias']
        num_features = self.get_feature_length()

        if self.cfg['setting'] not in ['LT Regression']:
            output_dim = self.num_class
        elif self.cfg['setting'] == 'LT Regression':
            output_dim = 1
        else:
            raise NotImplementedError

        if self.cfg['head']['type'] == "FCNorm":
            head = FCNorm(num_features, output_dim)
        elif self.cfg['head']['type']in ["FC", "cRT"]:
            head = nn.Linear(num_features, output_dim, bias=bias_flag)
        elif self.cfg['head']['type'] == "LWS":
            head = LWS(num_features, output_dim, bias=bias_flag)
        elif self.cfg['head']['type'] == "MLP":
            head = MLP(num_features, output_dim, self.cfg['head']['hidden_dims_lst'])
        else:
            raise NotImplementedError

        return head
    
    
    def get_feature_length(self):
        feat_len = 0
        for input_kw in self.input_kws:
            feat_len += self.model.config.get('hidden_dim_' + input_kw)
        return feat_len


    
