import os
import torch
from torchvision import transforms
from .vpr_model import VPRModel


class DINOV2SaladFeatureExtractor:

    def __init__(self, root, content):
        self.max_image_size = content["max_image_size"]
        self.device = "cuda" if content["cuda"] else "cpu"
        self.model = self.load_model(os.path.join(root, content["ckpt_path"]))
    
    def __call__(self, images):
        with torch.no_grad():
            scaled_imgs = self.downscale(images)
            b, c, h, w = scaled_imgs.shape
            h_new, w_new = (h // 14) * 14, (w // 14) * 14
            scaled_imgs = transforms.CenterCrop((h_new, w_new))(scaled_imgs)
            output = self.model(scaled_imgs).to(self.device)
            return output.cpu().numpy()
        
    def load_model(self, ckpt_path):
        model = VPRModel(
            backbone_arch='dinov2_vitb14',
            backbone_config={
                'num_trainable_blocks': 4,
                'return_token': True,
                'norm_layer': True,
            },
            agg_arch='SALAD',
            agg_config={
                'num_channels': 768,
                'num_clusters': 64,
                'cluster_dim': 128,
                'token_dim': 256,
            },
        )

        model.load_state_dict(torch.load(ckpt_path))
        model = model.eval().to(self.device)
        print(f"Loaded model from {ckpt_path} Successfully!")
        return model

    def downscale(self, images):
        if max(images.shape[-2:]) > self.max_image_size:
            b, c, h, w = images.shape
            # Maintain aspect ratio
            if h == max(images.shape[-2:]):
                w = int(w * self.max_image_size / h)
                h = self.max_image_size
            else:
                h = int(h * self.max_image_size / w)
                w = self.max_image_size
            return transforms.functional.resize(images, (h, w), interpolation=transforms.functional.InterpolationMode.BICUBIC)
        return images
    
    @property
    def feature_length(self):
        return 64 * 128 + 256    # num_clusters * cluster_dim + token_dim
