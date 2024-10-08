import cv2
import numpy as np
import os
import torch
from torchvision import transforms
from torchvision.transforms import functional
from .vpr_model import VPRModel


# Code from https://github.com/jacobgil/vit-explain/
def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = 0.7 * heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class DINOV2SaladFeatureExtractor:

    def __init__(self, root, content, pipeline=False, with_heatmap=True):
        self.max_image_size = content["max_image_size"]
        self.device = "cuda" if content["cuda"] else "cpu"
        self.saved_state, self.model = self.load_model(os.path.join(root, os.path.join(content["ckpt_path"],
                                    "model_best.pth") if pipeline else content["ckpt_path"]))
        self.with_heatmap = with_heatmap
    
    def __call__(self, images):
        resized_images = self.prepare_for_vit(images)
        encodings, descriptors, _, _ = self.model(resized_images)
        return encodings[0], descriptors
    
    def generate_heatmap(self, image, input_tensor):
        resized = self.prepare_for_vit(input_tensor).to(self.device)
        encodings, _, clst_feats, score_matrix = self.model(resized)
        h, w = encodings[0].shape[-2:]
        B, C = clst_feats.shape[:2]
        features_flat = clst_feats.view(B, C, -1)
        for c_i in range(C):
            features_flat[:, c_i, :] = torch.sum(features_flat[:, c_i, :] * score_matrix, dim=1)
        activations = torch.norm(features_flat, dim=1).reshape(h, w).cpu().numpy()
        activation_base = functional.to_pil_image(self.prepare_for_vit(functional.pil_to_tensor(image).unsqueeze(0)).squeeze())
        np_img = np.array(activation_base)[:, :, ::-1]
        mask = cv2.resize(1 - activations / np.max(activations), (np_img.shape[1], np_img.shape[0]))
        activations_map = show_mask_on_image(np_img, mask)
        return activations_map
        
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

        saved_state = torch.load(ckpt_path, map_location=self.device)
        if saved_state.keys() != {"epoch", "best_score", "state_dict"}:
            saved_state = {"epoch": 0, "best_score": 0, "state_dict": saved_state}
        # Remove module prefix from state dict
        state_dict_keys = list(saved_state["state_dict"].keys())
        for state_key in state_dict_keys:
            if state_key.startswith("module"):
                new_key = state_key.removeprefix("module.")
                saved_state["state_dict"][new_key] = saved_state["state_dict"][state_key]
                del saved_state["state_dict"][state_key]
        model.load_state_dict(saved_state["state_dict"])
        model = model.eval().to(self.device)
        print(f"Loaded model from {ckpt_path} successfully!")
        return saved_state, model

    def prepare_for_vit(self, images):
        images = self.downscale(images)
        b, c, h, w = images.shape
        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        return transforms.CenterCrop((h_new, w_new))(images)
    
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
    
    def torch_compile(self, **compile_args):
        self.model = torch.compile(self.model, **compile_args)
    
    def set_parallel(self):
        self.model = torch.nn.DataParallel(self.model)
    
    def set_float32(self):
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
