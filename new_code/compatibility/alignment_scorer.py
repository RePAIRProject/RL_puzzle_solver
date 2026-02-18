import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_erosion
from transformers import ViTModel, Dinov2Model
import torchvision.transforms as transforms


# ============================================================================
# Model Definition (copy from your models.py)
# ============================================================================
class MultiModalScorer(nn.Module):
    """RGB + Geometry + DINO semantic features."""

    def __init__(
        self,
        geometric_vit="google/vit-base-patch16-224",
        dino_model="facebook/dinov2-base",
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
            nn.Sigmoid(),
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

# ============================================================================
# Inference Helper
# ============================================================================

class AlignmentScorer:
    """Simple wrapper for inference."""
    
    def __init__(self, checkpoint_path, device='cuda', radius=30, threshold=30):
        self.device = device
        
        # Load model
        self.model = MultiModalScorer()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
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
    
    def score(self, rgb_images, mask_images):
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
        
        return scores.cpu().numpy().squeeze()
    
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
        from scipy.ndimage import distance_transform_edt, binary_erosion

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


# ============================================================================
# Usage Example
# ============================================================================

# if __name__ == '__main__':
#     # Load model
#     scorer = AlignmentScorer(
#         checkpoint_path='checkpoints/multimodal_best.pth',
#         device='cuda'
#     )
    
#     # Load your images
#     rgb_images = [
#         Image.open('alignment1.png').convert('RGB'),
#         Image.open('alignment2.png').convert('RGB'),
#         Image.open('alignment3.png').convert('RGB'),
#     ]
    
#     mask_images = [
#         Image.open('mask1.png').convert('L'),
#         Image.open('mask2.png').convert('L'),
#         Image.open('mask3.png').convert('L'),
#     ]
    
#     # Score them
#     scores = scorer.score(rgb_images, mask_images)
    
#     print(f"Scores: {scores}")
#     # Output: [0.95, 0.23, 0.87]