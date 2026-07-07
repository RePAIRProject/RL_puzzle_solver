import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_erosion
from compatibility.resnet_models import GuidedResNet, PairwiseCompatibilityModel
import torchvision.transforms as transforms


# ============================================================================
# Inference Helper
# ============================================================================

class AlignmentScorer:
    """Simple wrapper for inference."""
    
    def __init__(self, checkpoint_path, device='cuda', resnet_type='r50', radius=50, threshold=50, in_nc=3, guidance_nc=1):
        self.device = device

        # Load model
        self.encoder = GuidedResNet(variant=resnet_type, in_nc=in_nc, guidance_nc=guidance_nc)
        self.model = PairwiseCompatibilityModel(encoder=self.encoder)
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
        
        if not isinstance(mask_image, Image.Image):
            mask_image = Image.fromarray(mask_image)
        # if isinstance(mask_image, Image.Image):
        #     mask_array = np.array(mask_image)
        # else:
        #     mask_array = mask_image
        
        # scale if needed
        orig_w, orig_h = rgb_image.size
        if orig_w != 224 or orig_h != 224:
            rgb_resized = rgb_image.resize((224, 224), Image.BILINEAR)
        else:
            rgb_resized = rgb_image
        orig_wm, orig_hm = mask_image.size
        if orig_wm != 224 or orig_hm != 224:
            mask_resized = mask_image.resize((224, 224), Image.NEAREST)
            scale = 224.0 / max(orig_w, orig_h)
            mask_array = np.array(mask_resized)
        else:
            scale = 1 
            mask_array = np.array(mask_image)
        
        scaled_radius = max(1, int(round(self.radius * scale)))
        scaled_threshold = max(1, int(round(self.threshold * scale)))
        
        # Create geometric features
        geometric = self._create_geometric_features(mask_array, scaled_radius, scaled_threshold)
        
        # Transform
        rgb_tensor = self.transform(rgb_resized)
        geometric_tensor = torch.from_numpy(geometric).float()
        
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
            # the CNN scorer wants (image [3x224x224], guidance_map [1x224x224])
            geom_guidance_map = rgb_geom_tensor[5:6, :, :]
            rgb_geom_batch.append(geom_guidance_map)
        
        # Stack into batch
        rgb_batch = torch.stack(rgb_batch).to(self.device)
        rgb_geom_batch = torch.stack(rgb_geom_batch).to(self.device)
        
        # Inference
        with torch.no_grad():
            scores = self.model(rgb_batch, rgb_geom_batch)

        # ===================================
        # DEBUG VISUALIZATION
        # ===================================
        # np_scores = scores.cpu().numpy().squeeze()
        # np_comps = torch.sigmoid(scores).cpu().numpy().squeeze()
        # import matplotlib.pyplot as plt
        # num_images = len(rgb_images)
        # num_of_rows = np.ceil(num_images // 10).astype(np.uint8) + 1
        # num_of_columns = np.ceil(num_images // num_of_rows).astype(np.uint8) + 1
        # print(f"figure with {num_of_rows} rows and {num_of_columns} columns")
        # fig, axs = plt.subplots(num_of_rows, num_of_columns, figsize=(32,32))
        
        # fig.suptitle("Scores", fontsize=28)  
        # for i in range(len(rgb_batch)):
        #     row_idx = i // num_of_columns
        #     col_idx = i % num_of_columns
        #     # print(row_idx, i, len(np_scores))
        #     axs[row_idx, col_idx].set_title(f"S: {np_scores[i]:.02f}, CMP: {np_comps[i]:.03f}")
        #     axs[row_idx, col_idx].imshow(rgb_images[i])
        #     axs[row_idx, col_idx].set_xticks([])
        #     axs[row_idx, col_idx].set_yticks([])
        # plt.show()
        # breakpoint()
        
        return scores.cpu().numpy().squeeze()
    
    def _create_geometric_features(self, mask_array, radius, threshold):
        unique_values = np.unique(mask_array)
        unique_values = unique_values[unique_values > 0]
        if len(unique_values) < 2:
            return np.zeros((3, mask_array.shape[0], mask_array.shape[1]), dtype=np.float32)

        val_A, val_B = unique_values[0], unique_values[1]
        mask_A = mask_array == val_A
        mask_B = mask_array == val_B

        proximity_A = self._compute_proximity_inclusive(mask_A, mask_B)
        proximity_B = self._compute_proximity_inclusive(mask_B, mask_A)
        contact_strength = self._compute_contact_region_edge_based(mask_A, mask_B)

        return np.stack([proximity_A, proximity_B, contact_strength], axis=0).astype(np.float32)


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