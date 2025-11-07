import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Building blocks
# ---------------------------

class MultiScaleDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dw3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.dw5 = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels, bias=False)
        self.dw7 = nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(in_channels * 3)
        self.act = nn.ReLU(inplace=True)
        self.pw  = nn.Conv2d(in_channels * 3, out_channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.cat([self.dw3(x), self.dw5(x), self.dw7(x)], dim=1)
        x = self.act(self.bn(x))
        x = self.pw_bn(self.pw(x))
        return x


class LightweightTCNBlock(nn.Module):
    """Lightweight non-causal TCN block with depthwise-separable Conv1d + residual."""
    def __init__(self, channels, hidden=None, kernel_size=3, dilation=1, dropout=0.1, reduction=2):
        super().__init__()
        if hidden is None:
            hidden = max(channels // reduction, 8)

        pad = (kernel_size - 1) // 2 * dilation

        self.dwconv = nn.Conv1d(channels, channels, kernel_size,
                                padding=pad, dilation=dilation,
                                groups=channels, bias=False)   # depthwise
        self.pwconv1 = nn.Conv1d(channels, hidden, 1, bias=False)  # reduce
        self.pwconv2 = nn.Conv1d(hidden, channels, 1, bias=False)  # expand

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):  # [B, C, L]
        residual = x
        x = self.dwconv(x)               # [B, C, L]
        x = self.act(self.bn1(self.pwconv1(x)))  # bottleneck
        x = self.drop(x)
        x = self.bn2(self.pwconv2(x))
        x = self.drop(x)
        x = self.act(x + residual)
        return x


class LightweightCNNTCNStream(nn.Module):
    """Depthwise-separable CNN -> flatten spatial -> lightweight TCN -> reshape back."""
    def __init__(self, in_channels, out_channels, tcn_hidden=None, kernel_size=3, dropout=0.1, reduction=2):
        super().__init__()
        if tcn_hidden is None:
            tcn_hidden = max(out_channels // reduction, 16)

        # depthwise + pointwise CNN
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size,
                            padding=kernel_size//2, groups=in_channels, bias=False)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        # stack lightweight TCN blocks with increasing dilation
        self.tcn1 = LightweightTCNBlock(out_channels, tcn_hidden, kernel_size=3, dilation=1, dropout=dropout, reduction=reduction)
        self.tcn2 = LightweightTCNBlock(out_channels, tcn_hidden, kernel_size=3, dilation=2, dropout=dropout, reduction=reduction)
        self.tcn3 = LightweightTCNBlock(out_channels, tcn_hidden, kernel_size=3, dilation=4, dropout=dropout, reduction=reduction)

    def forward(self, x):  # [B, C, H, W]
        x = self.act(self.bn(self.pw(self.dw(x))))     # [B, Cout, H, W]
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)                        # sequence [B, C, L]
        x = self.tcn3(self.tcn2(self.tcn1(x)))         # [B, C, L]
        x = x.view(B, C, H, W)
        return x


# -------------------------------------------
# DWT Backbone (replacement)
# -------------------------------------------
# ---- Torch-native Haar DWT (differentiable) ----
def _haar_filters(device, dtype):
    inv_sqrt2 = 1.0 / (2.0 ** 0.5)
    h = torch.tensor([inv_sqrt2, inv_sqrt2], device=device, dtype=dtype)
    g = torch.tensor([inv_sqrt2, -inv_sqrt2], device=device, dtype=dtype)
    LL = torch.outer(h, h)
    LH = torch.outer(h, g)
    HL = torch.outer(g, h)
    HH = torch.outer(g, g)
    k = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)  # [4,1,2,2]
    return k

class UltraLightDWTStream(nn.Module):
    """
    Ultra-lightweight Haar DWT stream:
      - DWT expands channels x4
      - Minimal depthwise + projection
      - Fewer parameters than original version
    """
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        self.register_buffer("kernels", torch.empty(0), persistent=False)  # lazy init

        # keep a small mid channel if you want a bottleneck; optional
        mid_channels = max(out_channels // reduction, 4)

        self.post = nn.Sequential(
            # depthwise conv but use kernel_size=1 (lighter than 3x3)
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=1,
                      groups=in_channels * 4, bias=False),
            nn.BatchNorm2d(in_channels * 4),   # <- fixed: appropriate for [B,C,H,W]
            nn.ReLU(inplace=True),

            # single projection to out_channels
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),      # <- fixed
            nn.ReLU(inplace=True),
        )

    def _ensure_kernels(self, x):
        if self.kernels.numel() == 0 or self.kernels.device != x.device or self.kernels.dtype != x.dtype:
            k = _haar_filters(x.device, x.dtype)  # [4,1,2,2]
            # repeat for each input channel,
            # final weight shape [4*C, 1, 2, 2] is compatible with groups=C usage below
            self.kernels = k.repeat(self.in_channels, 1, 1, 1)  # [4*C,1,2,2]

    def forward(self, x):  # [B, C, H, W]
        self._ensure_kernels(x)
        B, C, H, W = x.shape
        # groups=C -> each input channel is convolved with its 4 filters producing 4C output channels
        y = F.conv2d(x, weight=self.kernels, stride=2, padding=0, groups=C)  # [B, 4C, H/2, W/2]
        return self.post(y)


# ----- Second-Order Pooling (returns [B, C, C]) -----
class SecondOrderPooling(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1)                      # [B, C, N]
        x = x - x.mean(dim=2, keepdim=True)
        cov = torch.bmm(x, x.transpose(1, 2))     # [B, C, C]
        cov = cov / (H * W)
        return cov

# ---- CBAM-like lightweight fusion
class AttentionFusion(nn.Module):
    def __init__(self, in_channels, out_dim=128, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sop = SecondOrderPooling()

        fusion_dim = in_channels * 2  # GAP (C) + summarized SOP (C)

        self.att_fc1 = nn.Linear(fusion_dim, max(fusion_dim // reduction, 4), bias=False)
        self.att_fc2 = nn.Linear(max(fusion_dim // reduction, 4), fusion_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.fc_out = nn.Linear(fusion_dim, out_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        gap_feat = self.gap(x).view(B, C)       # [B, C]
        sop_mat  = self.sop(x)                  # [B, C, C]
        sop_feat = sop_mat.mean(dim=-1)         # [B, C]

        fusion = torch.cat([gap_feat, sop_feat], dim=1)  # [B, 2C]

        att = self.att_fc1(fusion)
        att = F.relu(att, inplace=True)
        att = self.att_fc2(att)
        att = self.sigmoid(att)

        fusion = fusion * att
        return self.fc_out(fusion)  # [B, out_dim]


# ---------------------------
# 3-Stream Backbone
# ---------------------------

class MultiStreamWaveletNet(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, num_classes=2, dropout=0.2):
        super().__init__()
        # Stream 1: DWT projects to embed_dim channels (and halves H, W)
        self.dwt = UltraLightDWTStream(in_channels, embed_dim)              # [B, embed_dim, H/2, W/2]
        # Stream 2: Multi-scale depthwise works on embed_dim channels
        self.msdc = MultiScaleDepthwiseConv(embed_dim, embed_dim)
        # Stream 3: CNN+TCN also works on embed_dim channels
        self.cnn_tcn = LightweightCNNTCNStream(embed_dim, embed_dim, tcn_hidden=embed_dim//2, dropout=dropout)

        # Fusion on final feature map
        self.fusion = AttentionFusion(in_channels=embed_dim, out_dim=128, reduction=16)
        
        # Layer norm for fused embedding (applied to [B, 128])
        self.norm = nn.LayerNorm(128)

        # Single fused vector -> classifier
        self.classifier = nn.Linear(128, num_classes)
       
        self._init_weights()

    def forward(self, x, return_embedding=False):
        # Require even H, W for DWT
        assert x.shape[-1] % 2 == 0 and x.shape[-2] % 2 == 0, "Input H and W must be even for DWT."
        x1 = self.dwt(x)          # [B, embed_dim, H/2, W/2]
        x2 = self.msdc(x1)        # [B, embed_dim, H/2, W/2]
        x3 = self.cnn_tcn(x2)     # [B, embed_dim, H/2, W/2]
        fused = self.fusion(x3)   # [B, 128]

        # apply normalization on embedding before classifier
        fused = self.norm(fused)

        if return_embedding:
            return fused

        return self.classifier(fused)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
