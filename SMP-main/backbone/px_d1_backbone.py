import logging

import torch
import copy
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from backbone.vit_lora_d1 import VisionTransformer, PatchEmbed, Block, resolve_pretrained_cfg, build_model_with_cfg, \
    checkpoint_filter_fn, Attention_LoRA


class ViT_LoRA(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, n_tasks=10, rank=64):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         global_pool=global_pool,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                         representation_size=representation_size,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                         weight_init=weight_init, init_values=init_values,
                         embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn,
                         n_tasks=n_tasks, rank=rank)

    def forward(self, x, task_id, register_blk=-1, get_feat=False, get_cur_feat=False, get_cur_x=False,first_flow=0,scape=False,scape_params=None):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        # prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)
        for i, blk in enumerate(self.blocks):
            x = blk(x, task_id, register_blk == i, get_feat=get_feat, get_cur_feat=get_cur_feat,get_cur_x=get_cur_x,first_flow=first_flow,scape=scape,scape_params=scape_params)

        x = self.norm(x)

        return x

    def generator_combined_weight(self):
        for i, blk in enumerate(self.blocks):
            blk.attn.lora_weight_k_1 = nn.Linear(768, 768, bias=False)
            blk.attn.lora_weight_v_1 = nn.Linear(768, 768, bias=False)
            blk.attn.lora_weight_k_2 = nn.Linear(768, 768, bias=False)
            blk.attn.lora_weight_v_2 = nn.Linear(768, 768, bias=False)

            blk.attn.lora_weight_k_meg = nn.Parameter(torch.Tensor(768, 768))
            blk.attn.lora_weight_v_meg = nn.Parameter(torch.Tensor(768, 768))

            blk.attn.lora_weight_k_1.weight.data = copy.deepcopy(torch.mm(blk.attn.lora_B_k[0].weight.data, blk.attn.lora_A_k[0].weight.data))
            blk.attn.lora_weight_v_1.weight.data = copy.deepcopy(torch.mm(blk.attn.lora_B_v[0].weight.data, blk.attn.lora_A_v[0].weight.data))
            blk.attn.lora_weight_k_2.weight.data = copy.deepcopy(torch.mm(blk.attn.lora_B_k_2[0].weight.data, blk.attn.lora_A_k_2[0].weight.data))
            blk.attn.lora_weight_v_2.weight.data = copy.deepcopy(torch.mm(blk.attn.lora_B_v_2[0].weight.data, blk.attn.lora_A_v_2[0].weight.data))


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    # pretrained_cfg = resolve_pretrained_cfg(variant, kwargs=kwargs)
    pretrained_cfg = resolve_pretrained_cfg(variant)
    default_num_classes = pretrained_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = build_model_with_cfg(
        ViT_LoRA, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model

class SimpleVitNet(nn.Module):
    def __init__(self, args, pretrained):
        super(SimpleVitNet, self).__init__()
        print('This is for the BaseNet initialization.')
        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, n_tasks=1,
                            rank=args["rank"])
        self.backbone = _create_vision_transformer(args['backbone_type'], pretrained=True, **model_kwargs)

        for module in self.backbone.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()

        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"]
        self.protos_list_source = []
        self.covs_list_source = []

    def update_fc(self, nb_classes, nextperiod_initialization=None, sigma_value=0.,m=0.):
        fc = self.generate_fc(self.feature_dim, nb_classes,sigma=True, sigma_value=sigma_value, m=m).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            # fc.sigma.data = self.fc.sigma.data
            # margin = self.fc.margin.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
                # margin = torch.cat([margin, torch.zeros(nb_classes - nb_output).to(self._device)])
            fc.weight = nn.Parameter(weight)
            # fc.margin = nn.Parameter(margin)
        del self.fc
        self.fc = fc

    def update_proxy_fc(self, nb_classes,sigma_value=0.,m1=0.,m2=0.):
        self.proxy_fc = self.generate_fc(self.feature_dim, nb_classes, sigma=True,sigma_value=sigma_value, m=m1).to(self._device)
        self.proxy_fc2 = self.generate_fc(self.feature_dim, nb_classes,sigma=True,sigma_value=sigma_value, m=m2).to(self._device)
        self.proxy_fc2.weight.data = copy.deepcopy(self.proxy_fc.weight.data)

    def generate_fc(self, in_dim, out_dim, sigma=True,sigma_value=0., m=0.):
        fc = CosineLinear(in_dim, out_dim, sigma=sigma,sigma_value=sigma_value, m=m)
        return fc

    def replace_fc(self, dataset, data_loader):
        assert self.fc is not None

        protos_current, cov_current = self.extract_prototype(dataset, data_loader)
        self.fc.weight.data[-protos_current.shape[0]:, : ] = copy.deepcopy(F.normalize(protos_current,p=2,dim=1))
        # self.fc.margin.data[-protos_current.shape[0]:] = copy.deepcopy(self.proxy_fc.margin.data)
        self.fc.sigma.data = copy.deepcopy(self.proxy_fc.sigma.data)
        self.protos_list_source.append(protos_current)
        self.covs_list_source.append(cov_current)


    def extract_prototype(self, dataset, data_loader):
        with torch.no_grad():
            prog_bar = tqdm(data_loader)

            # extract embeddings
            embedding_list, label_list = [], []
            for _, batch in enumerate(prog_bar):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = self.forward_proto(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

                prog_bar.set_description('Extracting prototype from PTM =>')

            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)

            # construct prototype-based classifier
            class_list = sorted(np.unique(dataset.labels))
            proto_list = []
            cov_list = []
            for class_index in class_list:
                # calculate prototype
                data_index = (label_list == class_index).nonzero().squeeze(-1)
                embedding = embedding_list[data_index]
                proto = embedding.mean(0)
                cov = torch.cov(embedding.T) + torch.eye(proto.shape[-1]) * 1e-3
                proto_list.append(proto[None])
                cov_list.append(cov[None])

            return torch.cat(proto_list, dim=0), torch.cat(cov_list, dim=0)

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, test=False, label=None, first_flow=0):
        if not test:
            # Training
            x = self.backbone(x, 0, first_flow=first_flow)
            x = x[:, 0, :]
            if first_flow == 1:
                out = self.proxy_fc(x,label)
            else: # 0 表示训练时的测试，使用proxy_fc2
                out = self.proxy_fc2(x, label)
            out.update({"features": x})
            return out
        else:
            # Testing
            x = self.backbone(x, 0)
            x = x[:, 0, :]
            out = self.fc(x)
            out.update({"features": x})
            return out
    def forward_proto(self, x):
        x = self.backbone(x, 0)
        x = x[:, 0, :]
        return x
    def forward_scape(self, x, scape_params,only_features=False):
        x = self.backbone(x, 0,scape=True,scape_params=scape_params)
        x = x[:, 0, :]
        if only_features:
            return x
        else:
            out = self.fc(x)
            out.update({"features": x})
            return out
    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True,sigma_value=0., m=0.2):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        # self.sin_m = torch.sin(torch.tensor(m))  # 计算余量 m 的 sin 值
        # self.cos_m = torch.cos(torch.tensor(m))  # 计算余量 m 的 cos 值
        self.m = m
        self.sigma_value = sigma_value
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))

        if sigma:
            self.sigma = nn.Parameter(torch.tensor(self.sigma_value))
        else:
            self.register_parameter('sigma', None)

        self.reset_parameters()

        self.few_shot_weight = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        # if self.sigma is not None:
        #     self.sigma.data.fill_(1)

    def forward(self, input,label=None):
        if self.few_shot_weight is None:
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            cat_weight = torch.cat([self.weight,self.few_shot_weight],dim=0)
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(cat_weight, p=2, dim=1))

        # input_norm = input.detach().norm(dim=1, p=2, keepdim=True)
        # weight_norm = self.weight.detach().norm(dim=1, p=2, keepdim=True)
        # out = F.linear(input, self.weight) / (input_norm * weight_norm.T)

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)
        res = {}
        if ((self.m > 0.) or (self.m < 0.)) and label is not None:
            # ********************** v1 未优化 *************************
            # # 动态施加边距
            # sin_m = self.sin_m.to(self.weight.device)  # 计算余量 m 的 sin 值
            # cos_m = self.cos_m.to(self.weight.device)  # 计算余量 m 的 cos 值
            # one_hot = F.one_hot(label, num_classes=self.out_features)  # [batch_size, cout]
            # sin = (1 - out ** 2) ** 0.5  # 计算 sinθ
            # angle_sum = out * cos_m - sin * sin_m  # cos(θ + m)
            # out = angle_sum * one_hot + out * (1 - one_hot)  # 仅对正确类别应用余量
            # ********************** v2 高效版本 *************************
            pos = torch.gather(out, 1, label.view(-1, 1)) # cosθ
            new_pos = pos - self.m # cos(θ + m)
            out = torch.scatter(out, 1, label.view(-1, 1), new_pos)

        if self.sigma is not None:
            out_s = self.sigma * out
            res['logits'] = out_s

        return res

    def forward_few_shot_calib(self, input, few_shot_margin=None, label=None):
        if self.few_shot_weight is None:
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            cat_weight = torch.cat([self.weight,self.few_shot_weight],dim=0)
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(cat_weight, p=2, dim=1))

        # input_norm = input.detach().norm(dim=1, p=2, keepdim=True)
        # weight_norm = self.weight.detach().norm(dim=1, p=2, keepdim=True)
        # out = F.linear(input, self.weight) / (input_norm * weight_norm.T)

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)
        res = {}
        if few_shot_margin is not None and label is not None:
            # ********************** v1 未优化 *************************
            # # 动态施加边距
            # sin_m = self.sin_m.to(self.weight.device)  # 计算余量 m 的 sin 值
            # cos_m = self.cos_m.to(self.weight.device)  # 计算余量 m 的 cos 值
            # one_hot = F.one_hot(label, num_classes=self.out_features)  # [batch_size, cout]
            # sin = (1 - out ** 2) ** 0.5  # 计算 sinθ
            # angle_sum = out * cos_m - sin * sin_m  # cos(θ + m)
            # out = angle_sum * one_hot + out * (1 - one_hot)  # 仅对正确类别应用余量
            # ********************** v2 高效版本 *************************
            pos = torch.gather(out, 1, label.view(-1, 1)) # cosθ
            m = torch.gather(few_shot_margin, 0, label) # maigin
            new_pos = pos - m[:, None]
            out = torch.scatter(out, 1, label.view(-1, 1), new_pos)

        if self.sigma is not None:
            out_s = self.sigma * out
            res['logits'] = out_s

        return res

def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)