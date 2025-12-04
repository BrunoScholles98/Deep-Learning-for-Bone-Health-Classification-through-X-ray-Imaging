"""
M3T  –  Multi-plane & Multi-slice Transformer
=============================================
Implementação fiel ao paper (CVPR 2022) com:
  • 3-D CNN out_channels = 32
  • backbone 2-D = ResNet-50 pré-treinado
  • proj_dim / embed = 256
  • 8 camadas transformer, 8 heads
Correções:
  (1) positional embedding dinâmico
  (2) conv1 da ResNet aceita 1 ou 3 canais
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import repeat

# ---------------------------------------------------------------------------
# 3-D CNN
# ---------------------------------------------------------------------------
class CNN3DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 5, padding=2)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 5, padding=2)
        self.bn2   = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x

# ---------------------------------------------------------------------------
# 2-D CNN + projeção
# ---------------------------------------------------------------------------
class ExtractAndProject(nn.Module):
    def __init__(self, in_ch: int, proj_dim: int = 256):
        super().__init__()
        self.cnn2d = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # adapta conv1 p/ 1 ou 3 canais
        self.cnn2d.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        self.cnn2d.fc    = nn.Identity()           # saída 2048

        self.proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim)
        )

    def forward(self, x):                          # (B,C,D,H,W)
        B,C,D,H,W = x.shape
        # multi-plane/multi-slice
        Ecor = torch.cat(torch.split(x, 1, 2), 2)
        Esag = torch.cat(torch.split(x, 1, 3), 3)
        Eax  = torch.cat(torch.split(x, 1, 4), 4)

        Scor = (Ecor * x).permute(0,2,1,3,4)
        Ssag = (Esag * x).permute(0,3,1,2,4)
        Sax  = (Eax  * x).permute(0,4,1,2,3)

        S = torch.cat((Scor,Ssag,Sax), 1)          # (B,3N,C,H,W)
        S = S.reshape(-1,C,H,W)                    # (B·3N,C,H,W)

        feat = self.cnn2d(S)                       # (B·3N,2048)
        feat = self.proj(feat)                     # (B·3N,256)
        return feat.view(B, 3*H, -1)               # (B,3N,d)

# ---------------------------------------------------------------------------
# Embedding dinâmico
# ---------------------------------------------------------------------------
class EmbeddingLayer(nn.Module):
    def __init__(self, emb: int = 256):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1,1,emb))
        self.sep = nn.Parameter(torch.randn(1,1,emb))

        self.cor = nn.Parameter(torch.randn(1,emb))
        self.sag = nn.Parameter(torch.randn(1,emb))
        self.ax  = nn.Parameter(torch.randn(1,emb))

        self.pos = nn.Parameter(torch.randn(400, emb))  # 400 > 3*128+4

    def forward(self, t):                       # (B,3N,d)
        B, N3, D = t.shape
        cls = repeat(self.cls,'1 1 e -> b 1 e', b=B)
        sep = repeat(self.sep,'1 1 e -> b 1 e', b=B)

        x = torch.cat((
            cls, t[:,:128],  sep,
            t[:,128:256],    sep,
            t[:,256:],       sep
        ), 1)                                   # (B,L,d)

        x[:,:130]   += self.cor
        x[:,130:259]+= self.sag
        x[:,259:]   += self.ax

        x += self.pos[:x.size(1)]               # corta conforme L
        return x

# ---------------------------------------------------------------------------
# Transformer (8 camadas, 8 heads)
# ---------------------------------------------------------------------------
class MHSA(nn.Module):
    def __init__(self, emb=256, heads=8):
        super().__init__()
        self.h = heads; self.scale = emb**-0.5
        self.to_qkv = nn.Linear(emb, emb*3, bias=False)
        self.proj   = nn.Linear(emb, emb)

    def forward(self, x):
        qkv = self.to_qkv(x).reshape(x.size(0), x.size(1), 3, self.h, -1)
        q,k,v = qkv.unbind(2)
        att = (q @ k.transpose(-2,-1))*self.scale
        att = att.softmax(-1)
        out = (att @ v).transpose(1,2).reshape(x.size(0), x.size(1), -1)
        return self.proj(out)

class Residual(nn.Module):
    def __init__(self, fn): super().__init__(); self.fn=fn
    def forward(self,x): return x + self.fn(x)

class FFN(nn.Sequential):
    def __init__(self, emb): super().__init__(
        nn.Linear(emb, 3*emb), nn.GELU(), nn.Linear(3*emb, emb))

class Transformer(nn.Sequential):
    def __init__(self, depth=8, emb=256):
        layers=[]
        for _ in range(depth):
            layers += [
                Residual(nn.Sequential(nn.LayerNorm(emb), MHSA(emb))),
                Residual(nn.Sequential(nn.LayerNorm(emb), FFN(emb)))
            ]
        super().__init__(*layers)

# ---------------------------------------------------------------------------
# Classificador
# ---------------------------------------------------------------------------
class Classifier(nn.Module):
    def __init__(self, emb=256, n_cls=2):
        super().__init__(); self.fc = nn.Linear(emb, n_cls)
    def forward(self,x): return self.fc(x[:,0])

# ---------------------------------------------------------------------------
# Modelo completo
# ---------------------------------------------------------------------------
class M3T(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 out_ch: int = 32,
                 emb: int = 256,
                 depth: int = 8,
                 n_cls: int = 2):
        super().__init__()
        self.backbone = nn.Sequential(
            CNN3DBlock(in_ch, out_ch),
            ExtractAndProject(out_ch, emb),
            EmbeddingLayer(emb),
            Transformer(depth, emb)
        )
        self.head = Classifier(emb, n_cls)

    def forward(self, x):
        return self.head(self.backbone(x))

# ---------------------------------------------------------------------------
# quick check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B = 2; C = 3; D = H = W = 112
    dummy = torch.randn(B,C,D,H,W)
    print("output", M3T(in_ch=C)(dummy).shape)
