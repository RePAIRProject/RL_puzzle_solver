import torchvision.models.resnet as tvresnet
import torch, torch.nn as nn, torch.nn.functional as F



##############################################
#                                            #
#  ███████╗██████╗  █████╗ ██████╗ ███████╗  #
#  ██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝  #
#  ███████╗██████╔╝███████║██║  ██║█████╗    #
#  ╚════██║██╔═══╝ ██╔══██║██║  ██║██╔══╝    #
#  ███████║██║     ██║  ██║██████╔╝███████╗  #
#  ╚══════╝╚═╝     ╚═╝  ╚═╝╚═════╝ ╚══════╝  #
#                                            #
##############################################
# ---------------------------------------------------------------------------
# SPADE — Spatially-Adaptive Normalization
# ---------------------------------------------------------------------------

class SPADE(nn.Module):
    """
    Spatially-Adaptive (De)Normalization layer.

    Replaces the affine parameters of a normalization layer with spatially-
    varying gamma and beta predicted from a guidance map. The base norm step
    uses affine=False; ALL learned scale/shift comes from the guidance branch.

    Args:
        norm_nc      (int)  : channels in the feature map to be normalized.
        guidance_nc  (int)  : channels in the guidance map (1 for a float mask).
        norm_type    (str)  : 'group' (default) | 'instance' | 'batch'.
                              'group' is recommended for discriminative encoders —
                              batch-size independent and avoids cross-sample mixing.
        num_groups   (int)  : groups for GroupNorm (only used if norm_type='group').
                              norm_nc must be divisible by num_groups.
        hidden_nc    (int)  : intermediate channels in the guidance MLP.

    Forward:
        x        : [B, norm_nc, H, W]  — feature map to modulate.
        guidance : [B, guidance_nc, H_g, W_g]  — spatial guidance map.
                   Resized bilinearly to (H, W) inside forward().
    Returns:
        [B, norm_nc, H, W]  — modulated feature map.

    Usage note — (1 + gamma) vs gamma:
        The implementation uses out = x_norm * (1 + gamma) + beta.
        This is an initialisation trick confirmed by the original authors (issue #4):
        since conv weights start near zero, gamma ≈ 0 → (1 + gamma) ≈ 1,
        so SPADE is near-identity at init, which stabilises early training.
        Mathematically equivalent to plain gamma once training begins.
    """

    def __init__(
        self,
        norm_nc: int,
        guidance_nc: int = 1,
        norm_type: str = "instance",   # 'batch' | 'instance' | 'group'
        num_groups: int = 32,
        hidden_nc: int = 64,
    ):
        super().__init__()

        if norm_type == "group":
            # Batch-size independent; best for discriminative encoders.
            # Falls back gracefully when norm_nc < num_groups.
            # num_groups=32 is standard; norm_nc must be divisible by num_groups
            g = min(num_groups, norm_nc)
            assert norm_nc % g == 0, (
                f"norm_nc ({norm_nc}) must be divisible by num_groups ({g}). "
                f"Try num_groups=16 or a power of 2 that divides norm_nc."
            )
            self.norm = nn.GroupNorm(g, norm_nc, affine=False)

        elif norm_type == "instance":
            # Per-sample, no cross-sample mixing → cleanest for per-sample guidance.
            # Slightly weaker discriminative statistics than BN/GN.
            # Best for SPADE
            self.norm = nn.InstanceNorm2d(norm_nc, affine=False)

        elif norm_type == "batch":
            # Requires large batch >= ~32 for stable statistics.
            # Conceptually fights per-sample SPADE conditioning at small batches,
            # but offers strongest discriminative statistics
            # Use only if batch size is reliably large (>= 32) and GroupNorm
            # underperforms in ablations.
            self.norm = nn.BatchNorm2d(norm_nc, affine=False)

        else:
            raise ValueError(
                f"Unknown norm_type '{norm_type}'. Choose 'group', 'instance', or 'batch'."
            )

        # Lightweight guidance MLP: guidance_map → shared embedding → (gamma, beta)
        # Two 3×3 convs; kernel size 3 preserves spatial structure at all resolutions.
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(guidance_nc, hidden_nc, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(hidden_nc, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta  = nn.Conv2d(hidden_nc, norm_nc, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        # 1. Normalize (no learned affine — that comes entirely from guidance below)
        x_norm = self.norm(x)

        # 2. Resize guidance to match current feature spatial resolution
        if guidance.shape[2:] != x.shape[2:]:
            guidance = F.interpolate(
                guidance, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        # 3. Predict spatially-varying gamma and beta from guidance
        shared = self.mlp_shared(guidance)
        gamma  = self.mlp_gamma(shared)   # [B, norm_nc, H, W]
        beta   = self.mlp_beta(shared)    # [B, norm_nc, H, W]

        # 4. Modulate — (1 + gamma) initialises as near-identity; see docstring
        return x_norm * (1.0 + gamma) + beta




##########################################################
#                                                        #
#   ██████╗  █████╗ ████████╗███████╗██████╗             #
#  ██╔════╝ ██╔══██╗╚══██╔══╝██╔════╝██╔══██╗            #
#  ██║  ███╗███████║   ██║   █████╗  ██║  ██║            #
#  ██║   ██║██╔══██║   ██║   ██╔══╝  ██║  ██║            #
#  ╚██████╔╝██║  ██║   ██║   ███████╗██████╔╝            #
#   ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═════╝             #
#   ██████╗ ██████╗ ███╗   ██╗██╗   ██╗██████╗ ██████╗   #
#  ██╔════╝██╔═══██╗████╗  ██║██║   ██║╚════██╗██╔══██╗  #
#  ██║     ██║   ██║██╔██╗ ██║██║   ██║ █████╔╝██║  ██║  #
#  ██║     ██║   ██║██║╚██╗██║╚██╗ ██╔╝██╔═══╝ ██║  ██║  #
#  ╚██████╗╚██████╔╝██║ ╚████║ ╚████╔╝ ███████╗██████╔╝  #
#   ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝  ╚═══╝  ╚══════╝╚═════╝   #
#  ██████╗ ██╗      ██████╗  ██████╗██╗  ██╗███████╗     #
#  ██╔══██╗██║     ██╔═══██╗██╔════╝██║ ██╔╝██╔════╝     #
#  ██████╔╝██║     ██║   ██║██║     █████╔╝ ███████╗     #
#  ██╔══██╗██║     ██║   ██║██║     ██╔═██╗ ╚════██║     #
#  ██████╔╝███████╗╚██████╔╝╚██████╗██║  ██╗███████║     #
#  ╚═════╝ ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝     #
#                                                        #
##########################################################

# ---------------------------------------------------------------------------
# GuidanceGatedConv2d — guidance-conditioned soft-gated convolution
# ---------------------------------------------------------------------------

class GuidanceGatedConv2d(nn.Module):
    """
    Soft-gated convolution whose gate branch is conditioned on an external
    spatial guidance map (Yu et al., ICCV 2019 — extended with explicit guidance).

    Standard gated conv:  out = phi(W_f * x)  *  sigmoid(W_g * x)
    Guided  gated conv:   out = phi(W_f * x)  *  sigmoid(W_g * cat(x, guidance))

    The feature branch operates on x alone (no guidance) to preserve normal
    CNN semantics. The gate branch sees cat(x, guidance) so it learns where
    in the image the guidance says "pay attention" — and can refine that via
    the feature context — without polluting the feature computation itself.

    Args:
        in_nc        (int)  : input feature channels.
        out_nc       (int)  : output feature channels.
        guidance_nc  (int)  : guidance map channels (1 for a float mask).
        kernel_size  (int)  : conv kernel size (default 3).
        stride       (int)  : stride (default 1).
        padding      (int)  : padding (default 1, keeps spatial size for kernel 3).
        dilation     (int)  : dilation (default 1).
        activation          : activation on the feature branch (default ELU).
                              ELU is preferred over ReLU here: it has non-zero
                              gradient for negative inputs, helping the gate not
                              to kill information in low-guidance regions.

    Forward:
        x        : [B, in_nc, H, W]
        guidance : [B, guidance_nc, H_g, W_g]  — resized inside forward().
    Returns:
        [B, out_nc, H, W]
    """

    def __init__(
        self,
        in_nc: int,
        out_nc: int,
        guidance_nc: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        activation: nn.Module = None,
    ):
        super().__init__()

        self.activation = activation if activation is not None else nn.ELU(inplace=True)

        # Feature branch — sees only x
        self.feature_conv = nn.Conv2d(
            in_nc, out_nc, kernel_size, stride, padding, dilation
        )

        # Gate branch — sees cat(x, guidance), one extra input channel group
        self.gate_conv = nn.Conv2d(
            in_nc + guidance_nc, out_nc, kernel_size, stride, padding, dilation
        )

    def forward(self, x: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        # Resize guidance to match x if needed (e.g. after stride-2 pooling upstream)
        if guidance.shape[2:] != x.shape[2:]:
            guidance = F.interpolate(
                guidance, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        # Feature path
        features = self.activation(self.feature_conv(x))

        # Gate path: guidance informs where to open/close gates
        gate_input = torch.cat([x, guidance], dim=1)
        gate = torch.sigmoid(self.gate_conv(gate_input))

        return features * gate

# ---------------------------------------------------------------------------
# Adapted from Yu et al., ICCV 2019 — Free-Form Image Inpainting with Gated Convolution
# Reference impl: github.com/avalonstrel/GatedConvolution_pytorch
# ---------------------------------------------------------------------------
class GatedConv2d(nn.Module):
    """Standard gated conv (Yu et al. 2019) — gate from features only.
    Guidance-compatible signature: guidance is accepted but ignored."""
    def __init__(self, in_nc, out_nc, guidance_nc=None, kernel_size=3,
                 stride=1, padding=1, dilation=1, activation=nn.ELU(inplace=True)):
        super().__init__()
        self.feature_conv = nn.Conv2d(in_nc, out_nc, kernel_size,
                                      stride, padding, dilation)
        self.gate_conv    = nn.Conv2d(in_nc, out_nc, kernel_size,
                                      stride, padding, dilation)
        self.activation   = activation

    def forward(self, x, guidance=None):  # guidance ignored
        features = self.activation(self.feature_conv(x))
        gate     = torch.sigmoid(self.gate_conv(x))
        return features * gate

class PlainConv2d(nn.Module):
    """nn.Conv2d with a guidance-compatible signature (guidance is ignored).
    Allows all conv types to share forward(x, guidance) without conditionals."""
    def __init__(self, in_nc, out_nc, guidance_nc=None, kernel_size=3,
                 stride=1, padding=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_nc, out_nc, kernel_size,
                              stride=stride, padding=padding, **kwargs)

    def forward(self, x, guidance=None):
        return self.conv(x)   # guidance intentionally ignored



# #########################################################
#                                                       #
#   ██████╗ ██╗   ██╗██╗██████╗ ███████╗██████╗         #
#  ██╔════╝ ██║   ██║██║██╔══██╗██╔════╝██╔══██╗        #
#  ██║  ███╗██║   ██║██║██║  ██║█████╗  ██║  ██║        #
#  ██║   ██║██║   ██║██║██║  ██║██╔══╝  ██║  ██║        #
#  ╚██████╔╝╚██████╔╝██║██████╔╝███████╗██████╔╝        #
#   ╚═════╝  ╚═════╝ ╚═╝╚═════╝ ╚══════╝╚═════╝         #
#                                                       #
#  ██████╗ ███████╗███████╗███╗   ██╗███████╗████████╗  #
#  ██╔══██╗██╔════╝██╔════╝████╗  ██║██╔════╝╚══██╔══╝  #
#  ██████╔╝█████╗  ███████╗██╔██╗ ██║█████╗     ██║     #
#  ██╔══██╗██╔══╝  ╚════██║██║╚██╗██║██╔══╝     ██║     #
#  ██║  ██║███████╗███████║██║ ╚████║███████╗   ██║     #
#  ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝   ╚═╝     #
#                                                       #
#  ██████╗ ██╗      ██████╗  ██████╗██╗  ██╗███████╗    #
#  ██╔══██╗██║     ██╔═══██╗██╔════╝██║ ██╔╝██╔════╝    #
#  ██████╔╝██║     ██║   ██║██║     █████╔╝ ███████╗    #
#  ██╔══██╗██║     ██║   ██║██║     ██╔═██╗ ╚════██║    #
#  ██████╔╝███████╗╚██████╔╝╚██████╗██║  ██╗███████║    #
#  ╚═════╝ ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝    #
#                                                       #
#########################################################

# ---------------------------------------------------------------------------
# GuidedBasicBlock for shallow ResNets (18/34)
# which can use GatedConv2d or GuidanceGatedConv2d (default)
# ---------------------------------------------------------------------------
_CONV_TYPES = {
    'GuidanceGated': GuidanceGatedConv2d,
    'Gated':         GatedConv2d,
    'default':       PlainConv2d,
}

_NORM_TYPES = {
    'batch':    lambda nc, gnc: nn.BatchNorm2d(nc),
    'group':    lambda nc, gnc: nn.GroupNorm(min(32, nc), nc),
    'instance': lambda nc, gnc: nn.InstanceNorm2d(nc, affine=True),
    'spade':    lambda nc, gnc: SPADE(nc, guidance_nc=gnc),
}

class GuidedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 guidance_nc=1,
                 conv_block_type='GuidanceGated',
                 norm_type='batch'):
        super().__init__()
        ConvCls = _CONV_TYPES[conv_block_type]
        norm    = _NORM_TYPES[norm_type]

        self.conv1 = ConvCls(inplanes, planes, guidance_nc,
                             stride=stride, padding=1)
        self.norm1 = norm(planes, guidance_nc)

        self.conv2 = ConvCls(planes, planes, guidance_nc, padding=1)
        self.norm2 = norm(planes, guidance_nc)

        self.downsample = downsample
        self.model_ref  = None

        # remember whether norm needs guidance in forward
        self._norm_needs_guidance = (norm_type == 'spade')

    def _apply_norm(self, norm_layer, x, guidance):
        if self._norm_needs_guidance:
            return norm_layer(x, guidance)   # SPADE
        return norm_layer(x)                 # BN / GN / IN

    def forward(self, x):
        guidance = self.model_ref._guidance
        identity = x

        out = self.conv1(x,   guidance)
        out = self._apply_norm(self.norm1, out, guidance)
        out = F.relu(out, inplace=True)

        out = self.conv2(out, guidance)
        out = self._apply_norm(self.norm2, out, guidance)

        if self.downsample is not None:
            identity = self.downsample(x)

        return F.relu(out + identity, inplace=True)


# ---------------------------------------------------------------------------
# GuidedBottleneck for deeper ResNets (50/101)
# which can use GatedConv2d or GuidanceGatedConv2d (default)
# ---------------------------------------------------------------------------
class GuidedBottleneck(nn.Module):
    expansion = 4  # output = planes × 4; ResNet-50 ends at 512×4 = 2048 ch

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, guidance_nc=1, 
                 conv_block_type='GuidanceGated',
                 norm_type='batch'):
        super().__init__()
        ConvCls = _CONV_TYPES[conv_block_type]
        norm    = _NORM_TYPES[norm_type]

        # 1×1 — channel mixing, no spatial reasoning, no guidance
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        # 3×3 — spatial convolution, guidance goes HERE only
        self.gconv2 = ConvCls(
            planes, planes, guidance_nc, stride=stride, padding=1
        )
        self.bn2 = norm(planes, guidance_nc)

        # 1×1 — channel expansion, no guidance
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * 4)

        self.downsample = downsample
        self.model_ref  = None   # set by GuidedResNet

        # remember whether norm needs guidance in forward
        self._norm_needs_guidance = (norm_type == 'spade')

    def _apply_norm(self, norm_layer, x, guidance):
        if self._norm_needs_guidance:
            return norm_layer(x, guidance)   # SPADE
        return norm_layer(x)                 # BN / GN / IN

    def forward(self, x):
        guidance = self.model_ref._guidance
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        conv_out = self.gconv2(out, guidance)
        # apply_norm is needed to be able to apply SPADE normalization
        out = F.relu(self._apply_norm(self.bn2, conv_out, guidance), inplace=True)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return F.relu(out + identity, inplace=True)


#########################################################
#                                                       #
#  ██████╗ ███████╗███████╗███╗   ██╗███████╗████████╗  #
#  ██╔══██╗██╔════╝██╔════╝████╗  ██║██╔════╝╚══██╔══╝  #
#  ██████╔╝█████╗  ███████╗██╔██╗ ██║█████╗     ██║     #
#  ██╔══██╗██╔══╝  ╚════██║██║╚██╗██║██╔══╝     ██║     #
#  ██║  ██║███████╗███████║██║ ╚████║███████╗   ██║     #
#  ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝   ╚═╝     #
#                                                       #
#########################################################
class GuidedResNet(nn.Module):
    """
    Guided ResNet encoder configurable to any standard depth variant.

    variant   block_type       blocks_per_stage   embed_dim
    ───────────────────────────────────────────────────────
    'r18'     GuidedBasicBlock [2,2,2,2]           512
    'r34'     GuidedBasicBlock [3,4,6,3]           512
    'r50'     GuidedBottleneck [3,4,6,3]          2048
    'r101'    GuidedBottleneck [3,4,23,3]         2048
    """
    CONFIGS = {
        'r18':  (GuidedBasicBlock, [2, 2,  2, 2]),
        'r34':  (GuidedBasicBlock, [3, 4,  6, 3]),
        'r50':  (GuidedBottleneck, [3, 4,  6, 3]),
        'r101': (GuidedBottleneck, [3, 4, 23, 3]),
    }

    def __init__(self, variant='r34', in_nc=3, guidance_nc=1):
        super().__init__()
        self._guidance = None

        block_cls, stages = self.CONFIGS[variant]
        base = 64   # channels after stem — fixed across all variants

        self.stem_gconv = GuidanceGatedConv2d(in_nc, base, guidance_nc, kernel_size=7, stride=2, padding=3)
        self.stem_bn    = nn.BatchNorm2d(base)
        self.stem_pool  = nn.MaxPool2d(3, stride=2, padding=1)

        exp = block_cls.expansion   # 1 for BasicBlock, 4 for Bottleneck
        self.layer1 = self._make_stage(block_cls, base,       base,     stages[0], 1, guidance_nc)
        self.layer2 = self._make_stage(block_cls, base*exp,   base*2,   stages[1], 2, guidance_nc)
        self.layer3 = self._make_stage(block_cls, base*2*exp, base*4,   stages[2], 2, guidance_nc)
        self.layer4 = self._make_stage(block_cls, base*4*exp, base*8,   stages[3], 2, guidance_nc)

        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.embed_dim = base * 8 * exp   # 512 or 2048

        for m in self.modules():
            if isinstance(m, (GuidedBasicBlock, GuidedBottleneck)):
                object.__setattr__(m, 'model_ref', self)

    def _make_stage(self, block_cls, in_nc, planes, n_blocks, stride, gnc):
        out_nc = planes * block_cls.expansion
        downsample = None
        if stride != 1 or in_nc != out_nc:
            downsample = nn.Sequential(
                nn.Conv2d(in_nc, out_nc, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_nc),
            )
        layers = [block_cls(in_nc, planes, stride, downsample, gnc)]
        for _ in range(1, n_blocks):
            layers.append(block_cls(out_nc, planes, guidance_nc=gnc))
        return nn.Sequential(*layers)

    def forward(self, x, guidance):
        self._guidance = guidance
        f = self.stem_gconv(x, guidance)          # gated conv, sees full guidance detail
        f = F.relu(self.stem_bn(f), inplace=True)
        f = self.stem_pool(f)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        return self.pool(f).flatten(1)   # [B, embed_dim]



##################################################
#                                                #
#  ███████╗ ██████╗ ██████╗ ██████╗ ███████╗     #
#  ██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔════╝     #
#  ███████╗██║     ██║   ██║██████╔╝█████╗       #
#  ╚════██║██║     ██║   ██║██╔══██╗██╔══╝       #
#  ███████║╚██████╗╚██████╔╝██║  ██║███████╗     #
#  ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝     #
#                                                #
#  ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗       #
#  ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║       #
#  ██╔████╔██║██║   ██║██║  ██║█████╗  ██║       #
#  ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║       #
#  ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗  #
#  ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝  #
#                                                #
##################################################
# ---------------------------------------------------------------------------
# Compatibility Models
#
# 1) the classic `single` image version (RGB, guidance_map map)
# 2) the `Dual` split version (RGB_A, RGB_B, guidance_map)
# ---------------------------------------------------------------------------
class PairwiseCompatibilityModel(nn.Module):
    """
    Single-stream: joint image (both pieces arranged together) + contact map.
    The guidance does the work of directing attention to the seam.
    Simpler, fewer forward passes, easier to train.

    Input:
        x           : [B, 3, H, W]  — both pieces rendered on a single canvas
        contact_map : [B, 1, H, W]  — guidance, 0→ignore / 1→important
    """
    def __init__(self, encoder: nn.Module = None):
        super().__init__()
        if encoder is None:
            encoder = GuidedResNet()
        self.encoder = encoder
        self.head    = nn.Linear(encoder.embed_dim, 1)

    def forward(self, x, contact_map):
        embed = self.encoder(x, contact_map)   # [B, embed_dim]
        return self.head(embed)                # [B, 1] logit


class PairwiseCompatibilityDualModel(nn.Module):
    """
    Dual-stream (Siamese): each piece goes through the shared encoder separately,
    then embeddings are fused. The contact map guides both passes identically.
    More parameters in the head; forces piece-level representations.

    Input:
        patch_a     : [B, 3, H, W]
        patch_b     : [B, 3, H, W]
        contact_map : [B, 1, H, W]
    """
    def __init__(self, encoder: nn.Module = None, dropout=0.3):
        super().__init__()
        if encoder is None:
            encoder = GuidedResNet()
        self.encoder = encoder
        d = encoder.embed_dim
        self.head = nn.Sequential(
            nn.Linear(d * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, patch_a, patch_b, contact_map):
        ea = self.encoder(patch_a, contact_map)               # [B, d]
        eb = self.encoder(patch_b, contact_map)               # [B, d]
        fused = torch.cat([ea, eb, torch.abs(ea - eb)], dim=1)  # [B, d*3]
        return self.head(fused)                               # [B, 1] logit