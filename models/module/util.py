import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attention_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, attention_dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = Attention(dim=d_model, num_heads=nhead, attention_dropout=attention_dropout)
        self.pre_norm = nn.LayerNorm(d_model)
        self.linear1  = nn.Linear(d_model, d_model)
        self.linear2  = nn.Linear(d_model, d_model)
        self.norm1    = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.activation = F.gelu


    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.self_attn(self.pre_norm(src))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout1(src2)

        return src



class Transformer(nn.Module):
    def __init__(self,
                 sequence_length=None,
                 embedding_dim=64,
                 num_layers=1,
                 num_heads=1,
                 attention_dropout=0.,
                 mlp_dropout_rate=0.,
                 *args, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dropout=mlp_dropout_rate,
                                    attention_dropout=attention_dropout)
            for i in range(num_layers)])

        self.norm = nn.LayerNorm(embedding_dim)

        self.apply(self.init_weight)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


# FSRM Main model
class Attention_E(nn.Module):
    def __init__(self,
                 sequence_length=25,
                 embedding_dim=64,
                 *args, **kwargs):
        super(Attention_E, self).__init__()
        self.transformer = Transformer(sequence_length=sequence_length, embedding_dim=embedding_dim, *args, **kwargs)
        self.flattener = nn.Flatten(2, 3)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.flattener(x).transpose(-2, -1)
        x = self.transformer(x)
        x = x.transpose(-2, -1).reshape(b, c, h, w)
        return x

if __name__=="__main__":
    model = Attention_E(sequence_length=25, embedding_dim=640, num_layers=1)
    x = torch.randn(300, 640, 5, 5)
    y = model(x)
    print(y.shape)