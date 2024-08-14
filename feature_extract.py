from collections import OrderedDict
import os
import torch
from torchvision import transforms
from .vpr_model import VPRModel


class DINOV2SaladFeatureExtractor:

    def __init__(self, root, content, pipeline=False):
        self.max_image_size = content["max_image_size"]
        self.device = "cuda" if content["cuda"] else "cpu"
        self.saved_state, self.model = self.load_model(os.path.join(root, os.path.join(content["ckpt_path"],
                                    "model_best.pth") if pipeline else content["ckpt_path"]))
    
    def __call__(self, images):
        scaled_imgs = self.downscale(images)
        b, c, h, w = scaled_imgs.shape
        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        scaled_imgs = transforms.CenterCrop((h_new, w_new))(scaled_imgs)
        encodings, descriptors = self.model(scaled_imgs)
        return encodings[0], descriptors
        
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

        saved_state = torch.load(ckpt_path)
        if saved_state.keys() != {"epoch", "best_score", "state_dict"}:
            saved_state = {"epoch": 0, "best_score": 0, "state_dict": saved_state}
        model.load_state_dict(saved_state["state_dict"])
        model = model.eval().to(self.device)
        print(f"Loaded model from {ckpt_path} successfully!")
        return saved_state, model

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
    
    def set_train(self, is_train):
        self.model.train(is_train)
    
    def torch_compile(self, float32, **compile_args):
        self.model = torch.compile(self.model, **compile_args)
        if float32:
            self.model.to(torch.float32)
    
    def save_state(self, save_path, new_state):
        new_state["state_dict"] = self.model.state_dict()
        torch.save(new_state, save_path)
    
    @property
    def last_epoch(self): return self.saved_state["epoch"]

    @property
    def best_score(self): return self.saved_state["best_score"]

    @property
    def parameters(self): return self.model.parameters()
    
    @property
    def feature_length(self):
        return 64 * 128 + 256    # num_clusters * cluster_dim + token_dim
