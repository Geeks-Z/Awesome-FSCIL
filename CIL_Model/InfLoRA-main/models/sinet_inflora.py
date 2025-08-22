import torch
import torch.nn as nn
import copy
import math
from torch.nn import functional as F

from models.vit_inflora import VisionTransformer, PatchEmbed, Block,resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from models.zoo import CodaPrompt

class ViT_lora_co(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, n_tasks=10, rank=64):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn, n_tasks=n_tasks, rank=rank)


    def forward(self, x, task_id, register_blk=-1, get_feat=False, get_cur_feat=False):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)
        for i, blk in enumerate(self.blocks):
            x = blk(x, task_id, register_blk==i, get_feat=get_feat, get_cur_feat=get_cur_feat)

        x = self.norm(x)
        
        return x, prompt_loss



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
        ViT_lora_co, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model



class SiNet(nn.Module):

    def __init__(self, args):
        super(SiNet, self).__init__()

        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, n_tasks=args["total_sessions"], rank=args["rank"])
        self.image_encoder =_create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)
        # print(self.image_encoder)
        # exit()

        self.class_num = 1
        self.class_num = args["init_cls"]
        self.classifier_pool = nn.ModuleList([
            nn.Linear(args["embd_dim"], self.class_num, bias=True)
            for i in range(args["total_sessions"])
        ])

        self.classifier_pool_backup = nn.ModuleList([
            nn.Linear(args["embd_dim"], self.class_num, bias=True)
            for i in range(args["total_sessions"])
        ])

        # self.prompt_pool = CodaPrompt(args["embd_dim"], args["total_sessions"], args["prompt_param"])

        self.numtask = 0
        # Classifier heads for future tasks
        self.future_head = CosineLinear(768, args["nb_classes"])

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_vector(self, image, task=None):
        if task == None:
            image_features, _ = self.image_encoder(image, self.numtask-1)
        else:
            image_features, _ = self.image_encoder(image, task)
        image_features = image_features[:,0,:]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, image, get_feat=False, get_cur_feat=False, fc_only=False):
        if fc_only:
            fc_outs = []
            for ti in range(self.numtask):
                fc_out = self.classifier_pool[ti](image)
                fc_outs.append(fc_out)
            return torch.cat(fc_outs, dim=1)

        logits = []
        image_features, prompt_loss = self.image_encoder(image, task_id=self.numtask-1, get_feat=get_feat, get_cur_feat=get_cur_feat)
        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)
        for prompts in [self.classifier_pool[self.numtask-1]]:
            logits.append(prompts(image_features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features,
            'prompt_loss': prompt_loss
        }

    def interface(self, image, task_id = None):
        image_features, _ = self.image_encoder(image, task_id=self.numtask-1 if task_id is None else task_id)

        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)

        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        logits = torch.cat(logits,1)
        return logits
    def future_features(self, image, task_id = None):
        image_features, _ = self.image_encoder(image, task_id=self.numtask-1 if task_id is None else task_id)

        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)

        return image_features
    
    def interface1(self, image, task_ids):
        logits = []
        for index in range(len(task_ids)):
            image_features, _ = self.image_encoder(image[index:index+1], task_id=task_ids[index].item())
            image_features = image_features[:,0,:]
            image_features = image_features.view(image_features.size(0),-1)

            logits.append(self.classifier_pool_backup[task_ids[index].item()](image_features))

        logits = torch.cat(logits,0)
        return logits

    def interface2(self, image_features):

        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        logits = torch.cat(logits,1)
        return logits

    def update_fc(self, nb_classes):
        self.numtask +=1

    def classifier_backup(self, task_id):
        self.classifier_pool_backup[task_id].load_state_dict(self.classifier_pool[task_id].state_dict())

    def classifier_recall(self):
        self.classifier_pool.load_state_dict(self.old_state_dict)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}

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
