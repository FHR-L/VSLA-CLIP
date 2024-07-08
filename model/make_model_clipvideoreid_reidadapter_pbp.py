import torch
import torch.nn as nn
import numpy as np

from .clip.model import ResidualAttentionBlock, LayerNorm
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import einops

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num

        # hyperparameters
        self.use_learnable_prompt = cfg.MODEL.USE_LEARNABLE_PROMPT
        self.learnable_prompt_len = cfg.MODEL.PROMPT_LEN
        self.use_dat = cfg.MODEL.USE_ADAPTER

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size, cfg)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual
        self.cv_embed = None
        self.cv_embed_len = cfg.MODEL.PBP_PROMPT_LEN
        self.cv_embed_deep = cfg.MODEL.PBP_PROMPT_DEEP
        if cfg.MODEL.PBP_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(self.cv_embed_deep, self.cv_embed_len, camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)

        if self.use_learnable_prompt:
            print("use learnable prompt")
            self.prompt_learner = PromptLearner_Learnable(
                num_classes,
                clip_model.dtype,
                clip_model.token_embedding,
                self.learnable_prompt_len)
        else:
            print("not use learnable prompt")
            self.prompt_learner = PromptLearner_base(num_classes, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)



    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None):
        if get_text == True:
            prompts = self.prompt_learner(label) 
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        batch = x.shape[0]
        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x, use_dat=False)
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                img_feature_proj = image_features_proj[:, 0]
                img_feature_proj = einops.rearrange(img_feature_proj, '(b t) d -> b t d', b=batch)
                img_feature_proj = torch.mean(img_feature_proj, dim=1)
                return img_feature_proj

        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if self.cv_embed != None:
                cv_emb = self.cv_embed[:, :, cam_label]
            else:
                cv_emb = None
            image_features_last, image_features, image_features_proj = self.image_encoder(
                x,
                use_dat=self.use_dat,
                cv_emb=cv_emb)

            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]

        img_feature_last = einops.rearrange(img_feature_last, '(b t) d -> b t d', b=batch)
        img_feature_last = torch.mean(img_feature_last, dim=1)
        img_feature = einops.rearrange(img_feature, '(b t) d -> b t d', b=batch)
        img_feature = torch.mean(img_feature, dim=1)
        img_feature_proj = einops.rearrange(img_feature_proj, '(b t) d -> b t d', b=batch)
        img_feature_proj = torch.mean(img_feature_proj, dim=1)

        if self.training:
            feat = self.bottleneck(img_feature)
            feat_proj = self.bottleneck_proj(img_feature_proj)
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            return torch.cat([img_feature, img_feature_proj], dim=1)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))



def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from model.clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size, cfg):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    if cfg.MODEL.USE_VIFI_WEIGHT:
        state_dict = torch.load(cfg.MODEL.VIFI_WEIGHT, map_location="cpu")
        state_dict['token_embedding.weight'] = model.state_dict()['token_embedding.weight']
    model = clip.build_model(
        state_dict or model.state_dict(),
        h_resolution,
        w_resolution,
        vision_stride_size,
        seq_len=cfg.DATALOADER.SEQ_LEN,
        dat_type=cfg.MODEL.ADAPTER_TYPE
    )

    return model

class PromptLearner_Learnable(nn.Module):
    def __init__(self, num_class, dtype, token_embedding, prompt_len=0):
        super().__init__()
        if prompt_len == 0:
            assert 1 < 0
        self.prompt_len = prompt_len
        prefix_suffix_init = ''
        for i in range(self.prompt_len):
            prefix_suffix_init += "P "
        ctx_init = "X X X X "
        ctx_init = prefix_suffix_init + ctx_init + prefix_suffix_init
        ctx_init = ctx_init[:-1] + '.'

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda() # ([1, 77])
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor


        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, self.prompt_len + 1 + n_cls_ctx + self.prompt_len: , :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

        prefix_prompt = torch.empty(1, self.prompt_len, ctx_dim, dtype=dtype)
        suffix_prompt = torch.empty(1, self.prompt_len, ctx_dim, dtype=dtype)
        nn.init.normal_(prefix_prompt, std=0.02)
        nn.init.normal_(suffix_prompt, std=0.02)
        self.prefix_prompt = nn.Parameter(prefix_prompt)
        self.suffix_prompt = nn.Parameter(suffix_prompt)

    def forward(self, label):
        cls_ctx = self.cls_ctx[label] 
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1)
        prefix_prompt = self.prefix_prompt.expand(b, -1, -1)
        suffix_prompt = self.suffix_prompt.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                prefix_prompt,
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix_prompt,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts


class PromptLearner_Learnable2(nn.Module):
    def __init__(self, num_class, dtype, token_embedding, prompt_len=0):
        super().__init__()
        if prompt_len == 0:
            assert 1 < 0
        self.prompt_len = prompt_len
        suffix_init = ''
        for i in range(self.prompt_len):
            suffix_init += "P "
        ctx_init = "A photo of a X X X X person, "
        ctx_init = ctx_init + suffix_init
        ctx_init = ctx_init[:-1] + '.'

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")

        tokenized_prompts = clip.tokenize(ctx_init).cuda()  # ([1, 77])
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # [[49406, 320, 1125, 539, 320, 343, 343, 343, 343, 2533, 267, 335, 335, 335, 335, 335, 335, 335, 335, 269, 49407]]
        self.register_buffer("token_prefix", embedding[:, :5, :])
        self.register_buffer("token_middle", embedding[:, 9:11, :])
        self.register_buffer("token_suffix", embedding[:, self.prompt_len + 11:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx


        suffix_prompt = torch.empty(1, self.prompt_len, ctx_dim, dtype=dtype)
        nn.init.normal_(suffix_prompt, std=0.02)
        self.suffix_prompt = nn.Parameter(suffix_prompt)

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        middle = self.token_middle.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        suffix_prompt = self.suffix_prompt.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                middle,
                suffix_prompt,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts

class PromptLearner_base(nn.Module):
    def __init__(self, num_class, dtype, token_embedding):
        super().__init__()
        ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts