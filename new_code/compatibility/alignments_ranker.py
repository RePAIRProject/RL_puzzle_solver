import torch
import torch.nn as nn
from transformers import DINOv3ViTModel as Dinov2Model, ViTModel
import numpy as np 
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_erosion
import torchvision.transforms as transforms
import torch.nn.functional as F

"""
Because of the fact that we compute a combination of losses 
(at this moment BCE + custom Ranking Losses)
we will use both logits and scores (after sigmoid)

so all the models here will output the logits (before sigmoids) 
and whenever the model is used, sigmoid should be applied afterwards
"""

class BaselineScorer(nn.Module):
    """Baseline: Just RGB through ViT."""

    def __init__(self, pretrained_name="google/vit-base-patch16-224"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_name)

        # Simple scoring head
        self.scorer = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            # nn.Sigmoid(),
        )

    def forward(self, rgb):
        """
        Args:
            rgb: (B, 3, H, W) RGB images
        Returns:
            scores: (B, 1) alignment scores in [0, 1]
        """
        vit_feats = self.vit(rgb).pooler_output  # (B, 768)
        scores = self.scorer(vit_feats)
        return scores


class GeometricScorer(nn.Module):
    """RGB + 3 geometric channels."""

    def __init__(self, pretrained_name="google/vit-base-patch16-224"):
        super().__init__()

        # Project 6 channels to 3
        self.projection = nn.Conv2d(6, 3, kernel_size=1)

        # ViT backbone
        self.vit = ViTModel.from_pretrained(pretrained_name)

        # Scoring head
        self.scorer = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            # nn.Sigmoid(),
        )

    def forward(self, rgb_geometric):
        """
        Args:
            rgb_geometric: (B, 6, H, W)
                           channels 0-2: RGB
                           channels 3-5: geometric features
        Returns:
            scores: (B, 1)
        """
        x = self.projection(rgb_geometric)  # (B, 3, H, W)
        vit_feats = self.vit(x).pooler_output  # (B, 768)
        scores = self.scorer(vit_feats)
        return scores


class MultiModalScorer(nn.Module):
    """RGB + Geometry + DINO semantic features."""

    def __init__(
        self,
        geometric_vit="google/vit-base-patch16-224",
        dino_model="facebook/dinov3-vitb16-pretrain-lvd1689m",
    ):
        super().__init__()

        # Branch 1: Geometric ViT
        self.projection = nn.Conv2d(6, 3, kernel_size=1)
        self.geometric_vit = ViTModel.from_pretrained(geometric_vit)

        # Branch 2: DINO (frozen)
        self.dino = Dinov2Model.from_pretrained(dino_model)
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()
        print(f"DINO model loaded. Output dimension: {self.dino.config.hidden_size}")

        # Fusion head
        dino_dim = self.dino.config.hidden_size  # 768 for dinov2-base
        geom_dim = self.geometric_vit.config.hidden_size  # 768 for vit-base

        self.fusion = nn.Sequential(
            nn.Linear(geom_dim + dino_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            # nn.Sigmoid(),
        )

    def forward(self, rgb, rgb_geometric):
        """
        Args:
            rgb: (B, 3, H, W) - for DINO
            rgb_geometric: (B, 6, H, W) - for geometric branch
        Returns:
            scores: (B, 1)
        """
        # Geometric branch
        x = self.projection(rgb_geometric)
        geom_feats = self.geometric_vit(x).pooler_output  # (B, 768)

        # DINO branch (no gradients)
        with torch.no_grad():
            dino_feats = self.dino(rgb).pooler_output  # (B, 768)

        # Fuse
        combined = torch.cat([geom_feats, dino_feats], dim=1)  # (B, 1536)
        scores = self.fusion(combined)

        return scores


class CrossModalAttention(nn.Module):
    """Cross-attention between geometric and visual features."""

    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),  # Smaller expansion for small data
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        # Cross attention: query attends to key_value
        attended, _ = self.cross_attn(query, key_value, key_value)
        x = self.norm1(query + self.dropout(attended))

        # Feed-forward
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class MultiModalScorerV2(nn.Module):
    """
    Improved multi-modal scorer with:
    - Better geometric feature processing (Option B)
    - Wider fusion network (Option A)
    - Optional cross-attention (Option C)
    - Regularization for small data
    """

    def __init__(
        self,
        geometric_vit="google/vit-base-patch16-224",
        dino_model="facebook/dinov3-vitb16-pretrain-lvd1689m",
        use_cross_attention=False,
        dropout=0.2,
    ):
        """
        Args:
            geometric_vit: Pretrained ViT for geometric branch
            dino_model: Pretrained DINO for visual branch
            use_cross_attention: If True, use cross-modal attention (Option C)
            dropout: Dropout rate (higher for smaller datasets)
        """
        super().__init__()

        self.use_cross_attention = use_cross_attention

        # ============================================
        # Geometric Branch (Option B: Better Processing)
        # ============================================

        # Learn to process geometric features before ViT
        self.geometric_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
        )

        # Fuse RGB with encoded geometric features
        self.rgb_geom_fusion = nn.Sequential(
            nn.Conv2d(3 + 128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),
        )

        # ViT for geometric reasoning
        self.geometric_vit = ViTModel.from_pretrained(geometric_vit)

        # ============================================
        # Visual Branch (DINO - Frozen)
        # ============================================

        self.dino = Dinov2Model.from_pretrained(dino_model)
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()

        print(
            f"DINO model loaded (frozen). Output dimension: {self.dino.config.hidden_size}"
        )

        # ============================================
        # Cross-Modal Attention (Option C - Optional)
        # ============================================

        if use_cross_attention:
            self.geom_to_visual = CrossModalAttention(
                dim=768, num_heads=8, dropout=dropout
            )
            self.visual_to_geom = CrossModalAttention(
                dim=768, num_heads=8, dropout=dropout
            )
            print("Using cross-modal attention")

        # ============================================
        # Fusion Head (Option A: Wider, Better Regularization)
        # ============================================

        self.fusion = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            # nn.Linear(1024, 512),
            # nn.LayerNorm(512),
            # nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(1024, 256),
            # nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # Less dropout at end
            nn.Linear(256, 1),
            # No sigmoid - use BCEWithLogitsLoss
        )

    def forward(self, rgb, rgb_geometric):
        """
        Args:
            rgb: (B, 3, H, W) - RGB images for DINO
            rgb_geometric: (B, 6, H, W) - RGB + 3 geometric channels

        Returns:
            logits: (B, 1) - raw scores (apply sigmoid for probabilities)
        """
        # ============================================
        # Geometric Branch
        # ============================================

        # Split RGB and geometric channels
        rgb_only = rgb_geometric[:, :3]
        geom_only = rgb_geometric[:, 3:]

        # Encode geometric features
        geom_encoded = self.geometric_encoder(geom_only)  # (B, 128, H, W)

        # Fuse RGB with encoded geometric
        combined_input = torch.cat([rgb_only, geom_encoded], dim=1)  # (B, 131, H, W)
        vit_input = self.rgb_geom_fusion(combined_input)  # (B, 3, H, W)

        # Extract geometric features via ViT
        geom_output = self.geometric_vit(vit_input)
        geom_feats = geom_output.pooler_output  # (B, 768)

        # ============================================
        # Visual Branch (DINO)
        # ============================================

        with torch.no_grad():
            dino_output = self.dino(rgb)
            dino_feats = dino_output.pooler_output  # (B, 768)

        # ============================================
        # Cross-Modal Attention (Optional)
        # ============================================

        if self.use_cross_attention:
            # Add sequence dimension for attention
            geom_feats_seq = geom_feats.unsqueeze(1)  # (B, 1, 768)
            dino_feats_seq = dino_feats.unsqueeze(1)  # (B, 1, 768)

            # Bidirectional cross-attention
            geom_attended = self.geom_to_visual(geom_feats_seq, dino_feats_seq).squeeze(
                1
            )
            visual_attended = self.visual_to_geom(
                dino_feats_seq, geom_feats_seq
            ).squeeze(1)

            # Use attended features
            combined = torch.cat([geom_attended, visual_attended], dim=1)  # (B, 1536)
        else:
            # Simple concatenation
            combined = torch.cat([geom_feats, dino_feats], dim=1)  # (B, 1536)

        # ============================================
        # Final Fusion and Scoring
        # ============================================

        logits = self.fusion(combined)  # (B, 1)

        return logits

class MultiModalScorerV2_Practical(nn.Module):
    """
    Practical model for small datasets (~5K samples).
    
    Key features:
    - Frozen DINO (86M params)
    - Partially frozen ViT (14M trainable, 72M frozen)
    - Small new layers (2M params)
    - Total trainable: ~16M params
    """
    
    def __init__(self, 
                 geometric_vit='google/vit-base-patch16-224',
                 dino_model='facebook/dinov3-vitb16-pretrain-lvd1689m',
                 freeze_vit_layers=10,  # Freeze first 10 of 12 layers
                 dropout=0.4):
        super().__init__()
        
        # Geometric encoder (from scratch)
        self.geometric_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.rgb_geom_fusion = nn.Sequential(
            nn.Conv2d(3 + 128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1)
        )
        
        # ViT (partially frozen)
        self.geometric_vit = ViTModel.from_pretrained(geometric_vit)
        
        # Freeze early layers
        for name, param in self.geometric_vit.named_parameters():
            layer_num = self._extract_layer_num(name)
            if layer_num is not None and layer_num < freeze_vit_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        trainable_vit = sum(p.numel() for p in self.geometric_vit.parameters() if p.requires_grad)
        total_vit = sum(p.numel() for p in self.geometric_vit.parameters())
        print(f"ViT: {trainable_vit:,} trainable / {total_vit:,} total ({100*trainable_vit/total_vit:.1f}%)")
        
        # DINO (frozen)
        self.dino = Dinov2Model.from_pretrained(dino_model)
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()
        
        # Fusion (smaller for small data)
        self.fusion = nn.Sequential(
            nn.Linear(1536, 768),     # Smaller first layer
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 1)
        )
    
    def _extract_layer_num(self, param_name):
        """Extract layer number from parameter name."""
        import re
        match = re.search(r'encoder\.layer\.(\d+)', param_name)
        if match:
            return int(match.group(1))
        return None
    
    def forward(self, rgb, rgb_geometric):
        # Geometric processing
        rgb_only = rgb_geometric[:, :3]
        geom_only = rgb_geometric[:, 3:]
        geom_encoded = self.geometric_encoder(geom_only)
        combined_input = torch.cat([rgb_only, geom_encoded], dim=1)
        vit_input = self.rgb_geom_fusion(combined_input)
        
        # Feature extraction
        geom_feats = self.geometric_vit(vit_input).pooler_output
        
        with torch.no_grad():
            dino_feats = self.dino(rgb).pooler_output
        
        # Fusion
        combined = torch.cat([geom_feats, dino_feats], dim=1)
        logits = self.fusion(combined)
        
        return logits


class MultiModalScorerWeightedVit(nn.Module):
    """
    Multi-modal scorer that explicitly injects contact-region information
    into the ViT feature representation via weighted patch pooling.

    MultiModalScorerV2_Practical architecture flow:
        rgb_geometric → geometric_encoder → rgb_geom_fusion → ViT → CLS token (768-d)
                                                                           ↓
        rgb → DINO → DINO features (768-d)  →  concat → [CLS + DINO] (1536-d) → fusion head

    The CLS token is ViT's own aggregation of all patches via self-attention.
    It may or may not focus on the contact region.

    New architecture flow:
        rgb_geometric → geometric_encoder → rgb_geom_fusion → ViT → CLS token (768-d)
                                                               ↓
                                                     patch tokens (196 × 768)
                                                               ↓
                                                  contact-weighted pooling (768-d)
                                                                           ↓
        rgb → DINO → DINO features (768-d)  →  concat → [CLS + contact_pooled + DINO] (2304-d) → fusion head

    The contact channel (rgb_geometric[:, 5], shape B×H×W) is downsampled to the
    ViT patch grid (14×14 → 196 patches), flattened, and used as weights for a
    weighted mean over the 196 patch tokens. This produces a 768-d "contact-pooled"
    feature that explicitly captures what the boundary region looks like.

    Why this is better than an attention-correlation loss:
    - No output_attentions=True needed (no performance penalty)
    - Works with frozen ViT (pooling happens after the ViT)
    - Guaranteed signal (contact region always contributes)
    - Cleaner gradients (simple multiplication + averaging, no Pearson math)
    """

    def __init__(self,
                 geometric_vit='google/vit-base-patch16-224',
                 dino_model='facebook/dinov3-vitb16-pretrain-lvd1689m',
                 freeze_vit_layers=10,
                 dropout=0.4,
                 geometric_channel_scale=1.0):
        super().__init__()
        self.geometric_channel_scale = geometric_channel_scale

        # Geometric encoder (from scratch)
        self.geometric_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.rgb_geom_fusion = nn.Sequential(
            nn.Conv2d(3 + 128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1)
        )

        # ViT (partially frozen)
        self.geometric_vit = ViTModel.from_pretrained(geometric_vit)

        # Freeze early layers
        for name, param in self.geometric_vit.named_parameters():
            layer_num = self._extract_layer_num(name)
            if layer_num is not None and layer_num < freeze_vit_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True

        trainable_vit = sum(p.numel() for p in self.geometric_vit.parameters() if p.requires_grad)
        total_vit = sum(p.numel() for p in self.geometric_vit.parameters())
        print(f"ViT: {trainable_vit:,} trainable / {total_vit:,} total ({100*trainable_vit/total_vit:.1f}%)")

        # DINO (frozen)
        self.dino = Dinov2Model.from_pretrained(dino_model)
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()

        # Fusion head: input is CLS (768) + contact-pooled (768) + DINO (768) = 2304
        self.fusion = nn.Sequential(
            nn.Linear(2304, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 1)
        )

    def _extract_layer_num(self, param_name):
        import re
        match = re.search(r'encoder\.layer\.(\d+)', param_name)
        if match:
            return int(match.group(1))
        return None

    def _contact_weighted_pool(self, last_hidden_state, contact_mask):
        """
        Weighted pooling of ViT patch tokens using contact region as weights.

        Args:
            last_hidden_state: (B, 197, 768) from ViT — CLS + 196 patches
            contact_mask: (B, H, W) — channel 5 of rgb_geometric

        Returns:
            pooled: (B, 768) — weighted average of patch tokens
        """
        patch_tokens = last_hidden_state[:, 1:, :]  # (B, 196, 768)
        B, num_patches, dim = patch_tokens.shape
        grid_side = int(num_patches ** 0.5)  # 14

        contact_resized = F.interpolate(
            contact_mask.unsqueeze(1),  # (B, 1, H, W)
            size=(grid_side, grid_side),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)  # (B, 14, 14)

        weights = contact_resized.reshape(B, num_patches)  # (B, 196)
        weights = F.relu(weights) + 1e-6  # ensure positive
        weights_sum = weights.sum(dim=1, keepdim=True)  # (B, 1)

        pooled = torch.bmm(weights.unsqueeze(1), patch_tokens).squeeze(1) / weights_sum  # (B, 768)
        return pooled

    def forward(self, rgb, rgb_geometric):
        rgb_only = rgb_geometric[:, :3]
        geom_only = rgb_geometric[:, 3:] * self.geometric_channel_scale
        geom_encoded = self.geometric_encoder(geom_only)
        combined_input = torch.cat([rgb_only, geom_encoded], dim=1)
        vit_input = self.rgb_geom_fusion(combined_input)

        # Full ViT output with all patch tokens
        vit_output = self.geometric_vit(vit_input)
        cls_feats = vit_output.last_hidden_state[:, 0, :]  # (B, 768)

        # Contact-weighted pooling of patch tokens
        contact_mask = rgb_geometric[:, 5]  # (B, H, W)
        contact_pooled = self._contact_weighted_pool(vit_output.last_hidden_state, contact_mask)

        with torch.no_grad():
            dino_feats = self.dino(rgb).pooler_output

        combined = torch.cat([cls_feats, contact_pooled, dino_feats], dim=1)
        logits = self.fusion(combined)

        return logits

class FiLMViTBlock(nn.Module):
    """Wraps a single ViT block with FiLM conditioning."""
    def __init__(self, vit_block, t_dim=64, hidden_dim=768):
        super().__init__()
        self.block = vit_block
        self.film_mlp = nn.Sequential(
            nn.Linear(t_dim, 256),
            nn.GELU(),
            nn.Linear(256, hidden_dim * 2)
        )

    def forward(self, hidden_states, t_emb=None, head_mask=None, output_attentions=False):
        x = self.block(hidden_states, head_mask=head_mask)

        if t_emb is not None:
            gamma, beta = self.film_mlp(t_emb).chunk(2, dim=-1)
            x = gamma.unsqueeze(1) * x + beta.unsqueeze(1)

        return x


class MultiModalScorerWeightedViTFiLM(MultiModalScorerWeightedVit):
    """
    Multi-modal scorer with FiLM conditioning + contact-weighted pooling.

    Architecture:
        rgb_geometric → geometric_encoder → rgb_geom_fusion
            → FiLMViT ─┬→ CLS token
                        └→ patch tokens → contact-weighted pooling
                                                   ↓
        rgb → DINO → concat → fusion → score

    FiLM conditioning vector is computed by global average pooling of the
    geometric channels (proximity_A, proximity_B, contact) and projecting to t_dim.
    This tells each FiLM-wrapped ViT block what alignment configuration to expect,
    allowing dynamic modulation of self-attention and FFN computations.
    """

    def __init__(self, t_dim=64, film_layers=(8, 9, 10, 11), **kwargs):
        super().__init__(**kwargs)
        self.t_dim = t_dim
        self.film_layers = film_layers

        self.t_emb_proj = nn.Sequential(
            nn.Linear(3, t_dim),
            nn.GELU(),
        )

        for i in film_layers:
            self.geometric_vit.encoder.layer[i] = FiLMViTBlock(
                self.geometric_vit.encoder.layer[i], t_dim
            )

        trainable_film = sum(p.numel() for p in self.t_emb_proj.parameters())
        for i in film_layers:
            trainable_film += sum(p.numel() for p in self.geometric_vit.encoder.layer[i].film_mlp.parameters())
        print(f"FiLM: wrapped layers {film_layers}, t_dim={t_dim}, +{trainable_film:,} params")

    def forward(self, rgb, rgb_geometric):
        rgb_only = rgb_geometric[:, :3]
        geom_only = rgb_geometric[:, 3:] * self.geometric_channel_scale
        geom_encoded = self.geometric_encoder(geom_only)
        combined_input = torch.cat([rgb_only, geom_encoded], dim=1)
        vit_input = self.rgb_geom_fusion(combined_input)

        geom_pooled = rgb_geometric[:, 3:].mean(dim=[2, 3])  # (B, 3)
        t_emb = self.t_emb_proj(geom_pooled)                  # (B, t_dim)

        x = self.geometric_vit.embeddings(vit_input)
        for block in self.geometric_vit.encoder.layer:
            if isinstance(block, FiLMViTBlock):
                x = block(x, t_emb)
            else:
                x = block(x)
        x = self.geometric_vit.layernorm(x)  # (B, 197, 768)

        cls_feats = x[:, 0, :]
        contact_pooled = self._contact_weighted_pool(x, rgb_geometric[:, 5])

        with torch.no_grad():
            dino_feats = self.dino(rgb).pooler_output

        combined = torch.cat([cls_feats, contact_pooled, dino_feats], dim=1)
        logits = self.fusion(combined)

        return logits






####################################################################3

# ============================================================================
# Inference Helper
# ============================================================================

class AlignmentsRanker:
    """Simple wrapper for inference."""
    
    def __init__(self, checkpoint_path, device='cuda', radius=30, threshold=30, verbosity=1):
        self.device = device
        self.verbosity = verbosity

        # Load model
        self.model = MultiModalScorerWeightedViTFiLM()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if self.verbosity > 0:
            print(f"loading state dict from file {checkpoint_path}")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.verbosity > 0:
            print("loaded!")
        self.model = self.model.to(device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        self.radius = radius 
        self.threshold = threshold
        
    
    def preprocess(self, rgb_image, mask_image):
        """
        Preprocess one alignment image + mask.
        
        Args:
            rgb_image: PIL Image (RGB)
            mask_image: PIL Image (L) or numpy array with 2 unique non-zero values
        
        Returns:
            rgb: (3, 224, 224) tensor
            rgb_geometric: (6, 224, 224) tensor
        """
        # Ensure correct types
        if not isinstance(rgb_image, Image.Image):
            rgb_image = Image.fromarray(rgb_image)
        
        if isinstance(mask_image, Image.Image):
            mask_array = np.array(mask_image)
        else:
            mask_array = mask_image
        
        # Create geometric features
        geometric_features = self._create_geometric_features(mask_array)
        
        # Resize
        rgb_resized = rgb_image.resize((224, 224), Image.BILINEAR)
        geometric_resized = self._resize_geometric(geometric_features, (224, 224))
        
        # Transform
        rgb_tensor = self.transform(rgb_resized)
        geometric_tensor = torch.from_numpy(geometric_resized).float()
        
        rgb_geometric = torch.cat([rgb_tensor, geometric_tensor], dim=0)
        
        return rgb_tensor, rgb_geometric
    
    def score(self, rgb_images, mask_images, split_method='fast', batch_size_limit=8, top_k_tournament=2):
        import time
        """
        Score a batch of alignments.
        
        Args:
            rgb_images: List of PIL Images or single PIL Image
            mask_images: List of PIL Images/arrays or single PIL Image/array
        
        Returns:
            scores: numpy array of scores in [0, 1]
        """
        # Handle single image
        if isinstance(rgb_images, Image.Image):
            rgb_images = [rgb_images]
            mask_images = [mask_images]
        
        # Preprocess all
        rgb_batch = []
        rgb_geom_batch = []

        if len(rgb_images) > batch_size_limit:
            if split_method == 'fast':
                if self.verbosity > 1:
                    print('many images, using fast approximation!')
                time_t = time.time()
                scores, ranked_indices = self.fast_approximated_ranking_via_inference_batched(rgb_images, mask_images, batch_size=batch_size_limit)
                if self.verbosity > 1:
                    print(f'fast approximated ranking with {len(rgb_images)} took {(time.time() - time_t):.02f} seconds to rank!')
            elif split_method == 'tournament':
                # we need to split the images into smaller batches
                if self.verbosity > 1:
                    print('many images!, using tournament!')
                time_t = time.time()
                scores = self.inference_with_tournament(rgb_images, mask_images, batch_size=batch_size_limit, top_k=top_k_tournament)
                if self.verbosity > 1:
                    print(f'tournament with {len(rgb_images)} took {(time.time() - time_t):.02f} seconds to rank!')
            else:
                if self.verbosity > 0:
                    print(f'\n\n\nMethod `{split_method}` unknown! Not implmeneted!\n\n\n')
                breakpoint()
        else:
            for rgb_img, mask_img in zip(rgb_images, mask_images):
                rgb_tensor, rgb_geom_tensor = self.preprocess(rgb_img, mask_img)
                rgb_batch.append(rgb_tensor)
                rgb_geom_batch.append(rgb_geom_tensor)
            
            # Stack into batch
            rgb_batch = torch.stack(rgb_batch).to(self.device)
            rgb_geom_batch = torch.stack(rgb_geom_batch).to(self.device)
            
            # Inference
            with torch.no_grad():
                scores = self.model(rgb_batch, rgb_geom_batch)
                scores = scores.cpu().numpy().squeeze()
        
        return scores
    
    def inference_with_tournament(self, rgb_images, mask_images, batch_size=8, top_k=2):
        import matplotlib.pyplot as plt 
        """
        Tournament-style ranking for large image sets.
        Repeatedly scores batches of `batch_size`, keeps top `top_k` per batch,
        until <= batch_size images remain, then does a final ranking.
        
        Returns: final scores (numpy array) for the surviving images,
                and the indices of those images in the original input.
        """
        # Build index list to track original positions
        indices = list(range(len(rgb_images)))

        # Tournament rounds: keep reducing until <= batch_size images remain
        while len(indices) > batch_size:
            surviving_indices = []

            # Split current pool into batches of `batch_size`
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start : batch_start + batch_size]

                # If this tail batch is too small to be meaningful, carry it forward as-is
                if len(batch_indices) <= top_k:
                    surviving_indices.extend(batch_indices)
                    continue

                # Preprocess this batch
                rgb_batch, rgb_geom_batch = self._preprocess_batch(
                    [rgb_images[i] for i in batch_indices],
                    [mask_images[i] for i in batch_indices],
                )

                # Score
                with torch.no_grad():
                    scores = self.model(rgb_batch, rgb_geom_batch)
                scores = scores.cpu().numpy().squeeze()
                # comps = torch.sigmoid(torch.from_numpy(scores)) 

                # DBEUG
                # fig, axs = plt.subplots(1, len(rgb_batch))
                # fig.suptitle(f"batch {batch_start}, indices: {batch_indices}", fontsize=28)  
                # rgb_imgs = [rgb_images[i] for i in batch_indices]
                # for i in range(len(rgb_batch)):
                #     axs[i].set_title(f"score: {scores[i]:.03f}, comp: {comps.numpy()[i]:.03f}")
                #     axs[i].imshow(np.asarray(rgb_imgs[i]))
                # plt.show()
                # breakpoint()

                # Keep top_k from this batch
                top_local = np.argsort(scores)[::-1][:top_k]          # local indices
                surviving_indices.extend(batch_indices[local] for local in top_local)

            indices = surviving_indices  # shrunk pool for next round

        # --- Final round: score all remaining images together ---
        rgb_batch, rgb_geom_batch = self._preprocess_batch(
            [rgb_images[i] for i in indices],
            [mask_images[i] for i in indices],
        )
        with torch.no_grad():
            final_scores = self.model(rgb_batch, rgb_geom_batch)
        final_scores = final_scores.cpu().numpy().squeeze()
        full_scores = np.full(len(rgb_images), np.nan)
        full_scores[indices] = final_scores

        return full_scores #final_scores, indices


    def fast_approximated_ranking_via_inference_batched(self, rgb_images, mask_images, batch_size=8):
        """
        Score all images in batches of `batch_size`, then rank globally
        by concatenating all batch scores.

        Note: scores are not directly comparable across batches (see tournament
        version), but this gives a fast approximate ranking.

        Returns: global ranking as original indices (best first),
                and the full scores array (one score per input image).
        """
        # import matplotlib.pyplot as plt
        # breakpoint()
        # num_images = len(rgb_images)
        # num_of_rows = np.ceil(num_images // batch_size).astype(np.uint8) + 1
        # print(f"figure with {num_of_rows} rows and {batch_size} columns")
        # fig, axs = plt.subplots(num_of_rows, batch_size, figsize=(32,32))
        # fig.suptitle("Scores", fontsize=28)  
        all_scores = np.empty(len(rgb_images))

        for batch_start in range(0, len(rgb_images), batch_size):
            batch_slice = slice(batch_start, batch_start + batch_size)

            rgb_batch, rgb_geom_batch = self._preprocess_batch(
                rgb_images[batch_slice],
                mask_images[batch_slice],
            )

            with torch.no_grad():
                scores = self.model(rgb_batch, rgb_geom_batch)

            # comps = torch.sigmoid(torch.from_numpy(scores)) 
            # np_comps = torch.sigmoid(scores).cpu().numpy().squeeze()
            np_scores = scores.cpu().numpy().squeeze()
            all_scores[batch_slice] = np_scores #scores.cpu().numpy().squeeze()

            # np_images = rgb_batch.cpu().numpy().squeeze()
 
            if not isinstance(np_scores, np.ndarray):
                np_scores = np.asarray([np_scores])
            # row_idx = batch_start // batch_size

            # if len(rgb_batch) == 1:
            #     np_scores = [np_scores]

            # for i in range(np.minimum(len(rgb_batch), len(np_scores))):
            #     # print(row_idx, i, len(np_scores))
            #     axs[row_idx, i].set_title(f"S: {np_scores[i]:.02f}, CMP: {np_comps[i]:.03f}")
            #     axs[row_idx, i].imshow(rgb_images[batch_start+i])
            #     axs[row_idx, i].set_xticks([])
            #     axs[row_idx, i].set_yticks([])
            
        # plt.show()
        # breakpoint()

        ranked_indices = np.argsort(all_scores)[::-1]
        return all_scores, ranked_indices

    def _preprocess_batch(self, rgb_images, mask_images):
        """Preprocess a list of (rgb, mask) pairs into stacked tensors."""
        rgb_batch, rgb_geom_batch = [], []
        for rgb_img, mask_img in zip(rgb_images, mask_images):
            rgb_tensor, rgb_geom_tensor = self.preprocess(rgb_img, mask_img)
            rgb_batch.append(rgb_tensor)
            rgb_geom_batch.append(rgb_geom_tensor)
        return (
            torch.stack(rgb_batch).to(self.device),
            torch.stack(rgb_geom_batch).to(self.device),
        )

    def _create_geometric_features(self, mask_array):
        """
        Create 3 geometric feature channels.

        Returns:
            geometric: (3, H, W) numpy array
        """
        unique_values = np.unique(mask_array)
        unique_values = unique_values[unique_values > 0]

        if len(unique_values) < 2:
            return np.zeros((3, mask_array.shape[0], mask_array.shape[1]), dtype=np.float32)

        val_A = unique_values[0]
        val_B = unique_values[1]

        mask_A = mask_array == val_A
        mask_B = mask_array == val_B

        # Proximity channels (inclusive of piece interior)
        proximity_A = self._compute_proximity_inclusive(mask_A, mask_B)
        proximity_B = self._compute_proximity_inclusive(mask_B, mask_A)

        # Contact region
        contact_strength = self._compute_contact_region_edge_based(mask_A, mask_B)

        geometric = np.stack([proximity_A, proximity_B, contact_strength], axis=0)

        return geometric.astype(np.float32)

    def _compute_proximity_inclusive(self, mask, other_mask):
        """
        Proximity that includes pixels INSIDE the mask.

        Pixels inside the mask are considered "maximally close" to the piece.
        Pixels outside fade based on distance.

        Args:
            mask: Binary mask of the piece
            other_mask: Binary mask of the other piece (for overlap detection)
            radius: How far outside the mask proximity extends (in pixels)

        Returns:
            proximity: (H, W) array in [0, 1]
        """
        # Compute signed distance:
        # - Negative inside the mask
        # - Positive outside the mask
        # - Zero at the boundary

        from scipy.ndimage import distance_transform_edt

        radius = self.radius

        # Distance from outside to mask
        dist_outside = distance_transform_edt(~mask)

        # Distance from inside to boundary
        dist_inside = distance_transform_edt(mask)

        # Combine: negative inside, positive outside
        signed_distance = np.where(mask, -dist_inside, dist_outside)

        # Convert to proximity:
        # - Inside mask (negative distance): proximity = 1.0
        # - At boundary (distance = 0): proximity = 1.0
        # - Outside mask: proximity fades over `radius` pixels

        proximity = np.zeros_like(signed_distance, dtype=np.float32)

        # Inside the mask: full proximity
        proximity[mask] = 1.0

        # Outside the mask: fade linearly over `radius` pixels
        outside = ~mask
        proximity[outside] = np.clip(1.0 - (signed_distance[outside] / radius), 0, 1)

        # Handle overlap: pixels in both masks get max proximity
        overlap = mask & other_mask
        proximity[overlap] = 1.0

        return proximity
    
    def _compute_contact_region_edge_based(self, mask_A, mask_B):
        """
        Contact region based on edge-to-edge distance.

        For each pixel:
        - Compute distance to edge of piece A
        - Compute distance to edge of piece B
        - If both are small, it's in the contact region
        """
        # from scipy.ndimage import distance_transform_edt, binary_erosion

        threshold = self.threshold

        # Extract edges (boundaries) of each piece
        edge_A = mask_A & ~binary_erosion(mask_A)
        edge_B = mask_B & ~binary_erosion(mask_B)

        # Distance to nearest edge pixel
        dist_to_edge_A = distance_transform_edt(~edge_A)
        dist_to_edge_B = distance_transform_edt(~edge_B)

        # Contact region: close to both edges, but not inside either piece
        close_to_A = dist_to_edge_A < threshold
        close_to_B = dist_to_edge_B < threshold
        not_inside = (~mask_A) & (~mask_B)
        inside = mask_A & mask_B

        contact_region_outside_pieces = close_to_A & close_to_B & not_inside
        contact_region_anywhere = close_to_A & close_to_B | inside
        contact_region_inside_pieces = contact_region_anywhere ^ contact_region_outside_pieces

        # Convert to smooth strength
        contact_strength = contact_region_inside_pieces.astype(np.float32)

        # Optional: add smooth falloff
        combined_dist = dist_to_edge_A + dist_to_edge_B
        smooth_strength = np.maximum(0, threshold*2 - combined_dist) / (threshold*2)
        contact_strength = smooth_strength * contact_region_inside_pieces

        return np.clip(contact_strength, 0, 1)
    
    def _resize_geometric(self, geometric, target_size):
        """Resize geometric features."""
        resized = []
        for i in range(geometric.shape[0]):
            channel = Image.fromarray((geometric[i] * 255).astype(np.uint8))
            channel_resized = channel.resize((target_size[1], target_size[0]), Image.BILINEAR)
            resized.append(np.array(channel_resized).astype(np.float32) / 255.0)
        return np.stack(resized, axis=0)