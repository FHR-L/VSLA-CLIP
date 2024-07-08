import math

import torch.nn as nn
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim: int, num_mha_heads: int):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_mha_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, cls_embeds, prompt_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x num_texts x embed_dim
        """
        num_texts, _ = cls_embeds.shape
        # num_texts x embed_dim
        q = self.q_proj(cls_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        # num_heads x head_dim x num_texts
        q = q.permute(1,2,0)

        num_vids, num_frames, _ = prompt_embeds.shape
        # num_vids x num_frames x embed_dim
        k = self.k_proj(prompt_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0,2,1,3)

        # num_vids x num_frames x embed_dim
        v = self.v_proj(prompt_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0,2,3,1)

        # num_vids x num_heads x num_frames x num_texts
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)

        # num_vids x num_heads x head_dim x num_texts
        attention = v @ attention_weights
        # num_vids x num_texts x num_heads x head_dim
        attention = attention.permute(0,3,1,2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)

        # num_vids x num_texts x embed_dim
        o = self.out_proj(attention)
        return o


class FusionTransformer(nn.Module):
    def __init__(self, embed_dim, num_mha_heads, seq_len):
        super(FusionTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.attn = nn.MultiheadAttention(embed_dim, num_mha_heads)
        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, cls_embeds, prompt_embeds):
        bs = cls_embeds.size(0) // self.seq_len
        D = cls_embeds.size(-1)
        q = cls_embeds.reshape(bs, self.seq_len, D).transpose(0, 1)
        print(prompt_embeds.shape)
        assert 1 < 0
        kv = prompt_embeds.reshape(bs, self.seq_len * self.seq_len, D).transpose(0, 1)
        attn_output = self.attn(q, kv, kv)[0]
        out = (q + attn_output).transpose(0, 1).reshape(bs * self.seq_len, D)
        out = self.post_layernorm(out)

        return out

# class FusionTransformer(nn.Module):
#     def __init__(self, embed_dim, dropout, num_mha_heads):
#         super(FusionTransformer, self).__init__()
#         self.embed_dim = embed_dim
#         dropout = dropout
#
#         self.cross_attn = MultiHeadedAttention(embed_dim, num_mha_heads)
#
#         self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
#
#         self.layer_norm1 = nn.LayerNorm(self.embed_dim)
#         self.layer_norm2 = nn.LayerNorm(self.embed_dim)
#         self.layer_norm3 = nn.LayerNorm(self.embed_dim)
#         self.dropout = nn.Dropout(dropout)
#
#         self._init_parameters()
#
#     def _init_parameters(self):
#         for name, param in self.named_parameters():
#             if 'linear' in name or 'proj' in name:
#                 if 'weight' in name:
#                     nn.init.eye_(param)
#                 elif 'bias' in name:
#                     param.data.fill_(0.)
#
#     def forward(self, cls_embeds, prompt_embeds):
#         """
#         Input
#             text_embeds: num_texts x embed_dim
#             video_embeds: num_vids x num_frames x embed_dim
#         Output
#             out: num_vids x num_texts x embed_dim
#         """
#         cls_embeds = self.layer_norm1(cls_embeds)
#         prompt_embeds = self.layer_norm1(prompt_embeds)
#
#         # num_vids x num_texts x embed_dim
#         attn_out = self.cross_attn(cls_embeds, prompt_embeds)
#         attn_out = self.layer_norm2(attn_out)
#
#         linear_out = self.linear_proj(attn_out)
#         out = attn_out + self.dropout(linear_out)
#         out = self.layer_norm3(out)
#
#         return out