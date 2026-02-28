# hpsv2_scorer_diff.py  (replacement for your HPSv2Scorer)
import os
import torch
import torch.nn.functional as F
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

# CLIP mean/std (OpenAI)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

class HPSv2Scorer(torch.nn.Module): 
    """
    - Input: images (B,3,H,W) in either [0,1] or [-1,1] (auto-normalized)
    - Pure torch preprocessing (F.interpolate + (x-mean)/std)
    - No PIL/Processor/uint8 conversion -> preserves autograd connectivity
    """
    def __init__(self, dtype: torch.dtype, device: str):
        super().__init__()
        self.dtype  = dtype
        self.device = device

        # Load open_clip model
        self.model, _, _ = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision=dtype,
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        # Force checkpoint download
        hpsv2.score([], "")
        ckpt_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2_compressed.pt"
        state = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state['state_dict'])
        self.model.eval()  # eval mode but NOT wrapped in no_grad

        self.tokenizer = get_tokenizer('ViT-H-14')

        # Register mean/std
        mean = torch.tensor(_CLIP_MEAN, device=device, dtype=torch.float32).view(1,3,1,1)
        std  = torch.tensor(_CLIP_STD,  device=device, dtype=torch.float32).view(1,3,1,1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std",  std,  persistent=False)

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B,3,H,W) in [0,1] or [-1,1] (torch.Tensor, can require_grad)
        returns: (B,3,224,224) normalized, dtype=self.dtype, device=self.device
        """
        x = images
        # [-1,1] -> [0,1]
        if x.min() < 0:
            x = (x*0.5 + 0.5).clamp(0,1)
        x = x.to(torch.float32)  # do normalization in float32
        x = F.interpolate(x, size=(224,224), mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std
        # Cast to model dtype (graph is preserved)
        x = x.to(self.dtype, non_blocking=True).to(self.device, non_blocking=True)
        return x

    def forward(self, images: torch.Tensor, prompts):
        """
        images: (B,3,H,W) torch tensor (graph is preserved)
        prompts: List[str]
        returns: (B,) torch tensor (scores) with grad_fn
        """
        px = self._preprocess(images)
        text = self.tokenizer(prompts).to(self.device)

        out = self.model(px, text)  # output_dict=True provides image_features/text_features
        img_f = out["image_features"]  # (B,D)
        txt_f = out["text_features"]   # (B,D)

        # (Optional) normalize then cosine similarity
        img_f = torch.nn.functional.normalize(img_f, dim=-1)
        txt_f = torch.nn.functional.normalize(txt_f, dim=-1)

        logits = img_f @ txt_f.T             # (B,B)
        scores = torch.diagonal(logits)      # (B,)
        return scores