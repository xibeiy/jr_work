import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce


class MSA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: Rearrange('b n (h d) -> b h n d', h=self.heads)(t), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) / self.dim_head

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = Rearrange('b h n d -> b n (h d)')(out)

        return self.fc_out(out)


class MLP(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim)
        )

    def forward(self, x):
        return self.net(x)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_expansion=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.msa = MSA(dim, heads=heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, expansion=mlp_expansion)

    def forward(self, x):
        x = x + self.msa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, num_classes, dim=768, depth=12, heads=12, mlp_expansion=4):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_chans, dim)
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(dim, heads=heads, mlp_expansion=mlp_expansion)
            for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_blocks(x)
        x = self.ln(x[:, 0])
        x = self.head(x)
        return x
