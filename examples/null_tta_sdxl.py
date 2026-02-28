import os
import random
from typing import List, Tuple, Deque, Dict
import statistics
import csv
from collections import deque
import math  # For torch.pi
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline, DDIMScheduler, StableDiffusionXLPipeline
from diffusers import DDPMScheduler
from tqdm.auto import tqdm
from torch.utils.checkpoint import checkpoint # ✨ Verify import

# =========================
# Config
# =========================
device = "cuda"
dtype_pipe = torch.float16
# ✨ SDXL: change resolution
height, width = 1024, 1024
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# Default (overridden by args)
seed = 42

# Same prompt list as the v1.5 code
prompt_list = [
    "A passenger jet being serviced on a runway in an airport.",
    "Three people are preparing a meal in a small kitchen.",
    "A pair of planes parked in a small rural airfield.",
    "A bathroom with a stand alone shower and a peep window.",
    "Several vehicles with pieces of luggage on them with planes off to the side.",
    "a black motorcycle is parked by the side of the road",
    "A small bathroom with a tub, toilet, sink, and a laundry basket are shown.",
    "A bus stopped on the side of the road while people board it.",
    "A bunch of people posing with some bikes.",
    "a jet engine on the wing of a plane",
    "A bunch of bicycles parked on the street with items sitting around them",
    "A Dog standing in front of a doorway.",
    "Two small planes sitting near each other on a run way.",
    "there is a bus that has a bike attached to the front",
    "A bird that is sitting in the rim of a tire.",
    "The black motorcycle is parked on the sidewalk.",
    "A corner of a rest room with a big shower.",
    "a dog with a plate of food on the ground",
    "there is a very large plane that is stopped at the airport",
    "Bicycles with back packs parked in a public place.",
    "A white walled bathroom features beige appliances and furniture.",
    "Several bicycles sit parked nest to each other.",
    "Some big commercial planes all parked by each other.",
    "a woman holding a plate of cake in her hand",
    "yellow and red motorcycle with a man riding on it next to grass",
    "A motorcycle stands in front of three people on a sidewalk.",
    "classic cars on a city street with people and a dog",
    "People getting on a bus in the city",
    "A large commercial airliner silhoetted in the sun.",
    "Residential bathroom with modern design and tile floor.",
    "a bus with a view of a lot of traffic and the back of another bus with a billboard on the back end",
    "A young man riding through the air on top of a skateboard.",
    "A toy elephant is sitting inside a wooden car toy.",
    "A motorized bicycle covered with greens and beans.",
    "A Man sitting at a table in front of bowls of spices.",
    "there is a bathroom that has a lot of things on the floor",
    "A passenger jet aircraft flying in the sky.",
    "An eye level counter-view shows blue tile, a faucet, dish scrubbers, bowls, a squirt bottle and similar kitchen items.",
    "A TV sitting on top of a wooden stand.",
    "A person sitting on a motorcycle in the grass.",
    "A white toilet in a generic public bathroom stall.",
    "a couple of people in uniforms are sitting together",
    "A group of giraffe standing around each other.",
    "Street merchant with bowls of grains and other products.",
    "A man driving a luggage cart sitting on top of a runway.",
    "Residential bathroom with commode and shower and plain white walls.",
    "Ornate archway inset with matching fireplace in room.",
    "there is a red bus that has a mans face on it",
    "a wooden skate with a toy elephant inside of it",
    "a bunch of people on skiing on a hill"
]

negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, low quality, pixelated"

num_samples = 3
num_particles = 3
num_inference_steps = 100 
guidance_scale = 5.0      # ✨ SDXL: 7.5 -> 5.0

min_inner_steps = 5
max_inner_steps = 25
lr_uncond = 1e-2
tampering_coef = 0.008

DEFAULT_HYPERPARAM_TRIPLES = [
    (100, 0.002, 0.01)
]

base_log_dir_root_template = (
    "logs_sdxl/SDXL_NullTTA_particle_{num_particles}_tampering_steps_"
    "{min_inner_steps}_to_{max_inner_steps}_{seed}_target-{target}"
)


# =========================
# Utils
# =========================
def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    g = torch.Generator(device=device)
    g.manual_seed(s)
    return g


@torch.no_grad()
def decode_images(pipe: DiffusionPipeline, latents: torch.Tensor) -> torch.Tensor:
    # (for no_grad)
    vae = pipe.vae
    latents = latents.to(next(iter(vae.post_quant_conv.parameters())).dtype) 
    
    latents = latents / vae.config.scaling_factor
    
    # VAE may be in .train() mode (checkpointing), so call consistently even under no_grad
    try:
        if vae.training: 
             imgs = checkpoint(vae.decode, latents, use_reentrant=False).sample
        else:
             imgs = vae.decode(latents).sample
    except:
        imgs = vae.decode(latents).sample

    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    return imgs.float()


def decode_images_grad(pipe: DiffusionPipeline, latents: torch.Tensor) -> torch.Tensor:
    vae = pipe.vae
    
    # VAE is fp32 while latents may be fp16; cast to VAE dtype
    latents_fp32 = latents.to(vae.dtype)
    
    latents_fp32 = latents_fp32 / vae.config.scaling_factor

    # Reference: _decode in smc_sdxl.py
    imgs = checkpoint(vae.decode, latents_fp32, use_reentrant=False).sample 

    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    return imgs.float()


def tweedie_x0_from_eps(x_t: torch.Tensor, eps: torch.Tensor, alpha_t: torch.Tensor) -> torch.Tensor:
    a = alpha_t.to(device=x_t.device, dtype=x_t.dtype)
    if a.ndim == 0:
        a_view = a.view(1, 1, 1, 1)
    elif a.shape[0] == 1 and x_t.shape[0] > 1:
        a_view = a.view(1, 1, 1, 1)
    elif a.shape[0] == x_t.shape[0] and a.ndim == 1:
        a_view = a.view(-1, 1, 1, 1)
    elif a.shape[0] == x_t.shape[0] and a.ndim > 1:
        a_view = a
    else:
        a_view = a.view(1, 1, 1, 1)

    sqrt_one_minus_alpha = torch.sqrt(1.0 - a_view)
    sqrt_alpha = torch.sqrt(a_view)
    return (x_t - sqrt_one_minus_alpha * eps) / sqrt_alpha


# --- reward loaders (keep v1.5 style + target-only loading for memory) ---

class MockReward:
    def __call__(self, images, prompts):
        if isinstance(prompts, str):
            prompts = [prompts] * images.shape[0]
        # Return NaN to save memory
        return torch.tensor([float('nan')] * images.shape[0], device=images.device)

    def eval(self):
        pass

def get_pickscore_fn(target_name):
    # ✨ Memory optimization: do not load unless it's the target
    if target_name != "pickscore":
        return MockReward()

    try:
        import das.rewards as rewards
        reward_model = rewards.PickScore(device=device)
        print("Using PickScore reward.")
        return reward_model
    except Exception as e:
        print(f"Could not import das.rewards or load PickScore: {e}")
        return MockReward()


def get_aesthetic_fn(target_name):
    if target_name != "aesthetic":
        return MockReward()

    try:
        import das.rewards as rewards
        reward_model = rewards.aesthetic_score(device=device)
        print("Using aesthetic_score reward.")
        return reward_model
    except Exception as e:
        print(f"Could not import das.rewards or load aesthetic_score: {e}")
        return MockReward()


def get_hps_fn(target_name):
    if target_name != "hpsv2":
        return MockReward()

    try:
        import das.rewards as rewards
        reward_model = rewards.hps_score(device=device)
        print("Using hps_score reward.")
        return reward_model
    except Exception as e:
        print(f"Could not import das.rewards or load hps_score: {e}")
        return MockReward()


def get_imagereward_fn(target_name):
    if target_name != "imagereward":
        return MockReward()

    try:
        import das.rewards as rewards
        reward_model = rewards.ImageReward(device=device)
        print("Using ImageReward reward.")
        return reward_model
    except Exception as e:
        print(f"Could not import das.rewards or load ImageReward: {e}")
        return MockReward()


def save_image_tensor(img: torch.Tensor, path: str):
    if img.ndim == 4:
        img = img[0]
    arr = (img.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    Image.fromarray(arr).save(path)


def make_init_latents(pipe: DiffusionPipeline, h: int, w: int, batch_size: int, gen: torch.Generator):
    # ✨ SDXL latent size (128x128)
    lat_h, lat_w = h // 8, w // 8
    return torch.randn((batch_size, 4, lat_h, lat_w), generator=gen, device=pipe.device, dtype=pipe.unet.dtype)


# =========================
# Core optimizer (SDXL Adapted)
# =========================
class CFGOptWithBeamSearch:
    def __init__(
        self,
        pipe: DiffusionPipeline,
        guidance_scale: float,
        num_inference_steps: int,
        lr_uncond: float,
        min_inner_steps: int,
        max_inner_steps: int,
        lambda_alpha: float,
        lambda_beta: float,
        lambda_gamma: float,
        num_beams: int,
        tampering_coef: float = 0.008,
    ):
        self.pipe = pipe
        self.s = float(guidance_scale)
        self.T = int(num_inference_steps)
        self.lr_uncond = float(lr_uncond)
        self.min_inner_steps = int(min_inner_steps)
        self.max_inner_steps = int(max_inner_steps)
        self.lambda_alpha = float(lambda_alpha)
        self.lambda_reg = float(lambda_beta)
        self.phi_variance = float(lambda_gamma)
        self.K_samples = int(num_beams)
        self.tampering_coef = float(tampering_coef)

        # Freeze all models
        for p in self.pipe.unet.parameters():
            p.requires_grad_(False)
        if hasattr(self.pipe, "vae"):
            for p in self.pipe.vae.parameters():
                p.requires_grad_(False)
        if hasattr(self.pipe, "text_encoder"):
            for p in self.pipe.text_encoder.parameters():
                p.requires_grad_(False)
        if hasattr(self.pipe, "text_encoder_2"):
            for p in self.pipe.text_encoder_2.parameters():
                p.requires_grad_(False)

        if hasattr(self.pipe, "vae"):
            self.pipe.vae.enable_gradient_checkpointing()
            self.pipe.vae.train() # checkpointing works only in train mode


    def optimize(
        self,
        prompt: str,
        # ✨ SDXL: Embeddings passed from outside
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        negative_pooled_prompt_embeds: torch.Tensor,
        add_time_ids: torch.Tensor,
        negative_add_time_ids: torch.Tensor,
        # ---
        reward_fn, 
        x_T_init_batch: torch.Tensor,
    ) -> torch.Tensor:

        K = self.K_samples
        if x_T_init_batch.shape[0] != K:
            print(f"Warning: x_T_init_batch size ({x_T_init_batch.shape[0]}) != K ({K}). Using first particle.")
            x_T_init_batch = x_T_init_batch[0:1].repeat(K, 1, 1, 1)

        # ✨ SDXL: Optimization targets
        un_orig = negative_prompt_embeds.clone()
        un_master = un_orig.detach().float().clone().requires_grad_(True)
        
        pooled_un_orig = negative_pooled_prompt_embeds.clone()
        pooled_un_master = pooled_un_orig.detach().float().clone().requires_grad_(True)

        cond_emb = prompt_embeds.clone()
        pooled_cond_emb = pooled_prompt_embeds.clone()

        s = self.pipe.scheduler
        timesteps = s.timesteps
        alphas = s.alphas_cumprod.to(self.pipe.device)
        betas = s.betas.to(self.pipe.device)

        do_classifier_free_guidance = self.s > 1.0

        # (always use checkpointing regardless of grad/no_grad)
        def call_unet_sdxl(latents_in, t_in, embed_main, embed_pooled, time_ids):
            added_cond_kwargs = {"text_embeds": embed_pooled, "time_ids": time_ids}
            
            # Use use_reentrant=False as in the SMC code
            noise_pred_tuple = checkpoint(
                self.pipe.unet,
                latents_in, t_in, embed_main,
                None, None, None, None, 
                added_cond_kwargs,
                use_reentrant=False # SMC/DNO reference style
            )
            return noise_pred_tuple[0]
        # ---------------------------------

        # initial selection
        with torch.no_grad():
            t_start = timesteps[0]
            t_tensor_start = torch.tensor([int(t_start)], device=x_T_init_batch.device, dtype=torch.long)
            a_t_start = alphas[int(t_start)]

            lat_in = torch.cat([x_T_init_batch] * 2) if do_classifier_free_guidance else x_T_init_batch
            lat_in = self.pipe.scheduler.scale_model_input(lat_in, t_start)
            t_in = t_tensor_start.repeat(K * 2) if do_classifier_free_guidance else t_tensor_start.repeat(K)

            uc_main = un_orig.repeat(K, 1, 1)
            uc_pool = pooled_un_orig.repeat(K, 1)
            uc_time = negative_add_time_ids.repeat(K, 1)
            c_main = cond_emb.repeat(K, 1, 1)
            c_pool = pooled_cond_emb.repeat(K, 1)
            c_time = add_time_ids.repeat(K, 1)

            if do_classifier_free_guidance:
                emb_main = torch.cat([uc_main, c_main])
                emb_pool = torch.cat([uc_pool, c_pool])
                emb_time = torch.cat([uc_time, c_time])
            else:
                emb_main = c_main
                emb_pool = c_pool
                emb_time = c_time

            noise_pred_start = call_unet_sdxl(lat_in, t_in, emb_main, emb_pool, emb_time)

            if do_classifier_free_guidance:
                noise_pred_uncond_start, noise_pred_text_start = noise_pred_start.chunk(2)
                eps_cfg_start = noise_pred_uncond_start + self.s * (noise_pred_text_start - noise_pred_uncond_start)
            else:
                eps_cfg_start = noise_pred_start

            x0_hat_start = tweedie_x0_from_eps(x_T_init_batch, eps_cfg_start, a_t_start)
            imgs_start = decode_images(self.pipe, x0_hat_start)
            
            scores_start = reward_fn(imgs_start, [prompt] * K)
            scores_start[torch.isnan(scores_start)] = -float('inf')

            _, best_idx_start = torch.max(scores_start, dim=0)
            x_t_particle = x_T_init_batch[best_idx_start:best_idx_start + 1].clone()


        for i, t in enumerate(tqdm(timesteps, desc="Null-TTA Optimization", leave=False)):
            t_int = int(t)
            t_tensor = torch.tensor([t_int], device=x_t_particle.device, dtype=torch.long)
            a_t_bar = alphas[t_int]

            # 1. Baseline eps_u (original uncond)
            with torch.no_grad():
                lat_in = torch.cat([x_t_particle] * 2) if do_classifier_free_guidance else x_t_particle
                lat_in = self.pipe.scheduler.scale_model_input(lat_in, t)
                t_in = t_tensor.repeat(2) if do_classifier_free_guidance else t_tensor

                if do_classifier_free_guidance:
                    emb_main = torch.cat([un_orig, cond_emb])
                    emb_pool = torch.cat([pooled_un_orig, pooled_cond_emb])
                    emb_time = torch.cat([negative_add_time_ids, add_time_ids])
                else:
                    emb_main = un_orig
                    emb_pool = pooled_un_orig
                    emb_time = negative_add_time_ids

                noise_pred_orig = call_unet_sdxl(lat_in, t_in, emb_main, emb_pool, emb_time)

                if do_classifier_free_guidance:
                    noise_pred_uncond_orig, _ = noise_pred_orig.chunk(2)
                    eps_u_orig_particle = noise_pred_uncond_orig
                else:
                    eps_u_orig_particle = noise_pred_orig

            # 2. Inner optimization loop
            tampering_progress = min(1.0, (1.0 + self.tampering_coef) ** i - 1.0)
            current_inner_steps = self.min_inner_steps + (self.max_inner_steps - self.min_inner_steps) * tampering_progress
            current_inner_steps_int = int(round(current_inner_steps))

            opt = torch.optim.Adam([un_master, pooled_un_master], lr=self.lr_uncond)

            with torch.cuda.amp.autocast(dtype=torch.float32):
                for inner_idx in range(current_inner_steps_int):
                    opt.zero_grad(set_to_none=True)

                    un_cur = un_master.to(self.pipe.unet.dtype)
                    pooled_un_cur = pooled_un_master.to(self.pipe.unet.dtype)

                    lat_in = self.pipe.scheduler.scale_model_input(x_t_particle, t_int)
                    lat_in_batch = torch.cat([lat_in]*2) if do_classifier_free_guidance else lat_in
                    t_in_batch = t_tensor.repeat(2) if do_classifier_free_guidance else t_tensor

                    if do_classifier_free_guidance:
                        emb_main = torch.cat([un_cur, cond_emb])
                        emb_pool = torch.cat([pooled_un_cur, pooled_cond_emb])
                        emb_time = torch.cat([negative_add_time_ids, add_time_ids])
                    else:
                        emb_main = un_cur
                        emb_pool = pooled_un_cur
                        emb_time = negative_add_time_ids
                    
                    # UNet in fp16, VAE/Reward in fp32
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        noise_pred_reward = call_unet_sdxl(lat_in_batch, t_in_batch, emb_main, emb_pool, emb_time)

                        if do_classifier_free_guidance:
                            eps_u_cur, eps_c_cur = noise_pred_reward.chunk(2)
                            eps_cfg_cur = eps_u_cur + self.s * (eps_c_cur - eps_u_cur)
                        else:
                            eps_u_cur = noise_pred_reward
                            eps_cfg_cur = noise_pred_reward
                    
                    try:
                        x0_hat = tweedie_x0_from_eps(x_t_particle, eps_cfg_cur.float(), a_t_bar.view(-1, 1, 1, 1))
                        imgs_reward = decode_images_grad(self.pipe, x0_hat) # VAE (fp32)
                        R_val = reward_fn(imgs_reward, [prompt]) # Reward (fp32)
                        R_val[torch.isnan(R_val)] = 0.0
                        reward_loss = R_val[0]
                    except Exception as e:
                        print(f"Error in VAE/Reward forward: {e}")
                        reward_loss = torch.tensor(0.0, device=device)

                    beta_t = betas[t_int]
                    alpha_bar_t = alphas[t_int]
                    denom = 2 * (1-beta_t) * (1-alpha_bar_t)
                    lambda_t = (beta_t / denom) if denom > 1e-9 else 0.0
                    
                    eps_loss = F.mse_loss(eps_u_cur.float(), eps_u_orig_particle.detach().float(), reduction='sum')
                    weighted_eps_loss = lambda_t * eps_loss

                    emb_reg = F.mse_loss(un_master, un_orig.float(), reduction='sum')
                    pool_reg = F.mse_loss(pooled_un_master, pooled_un_orig.float(), reduction='sum')
                    total_emb_reg = emb_reg + pool_reg

                    sig_coeff = self.lambda_reg * (1.0 / (2.0 * self.phi_variance + 1e-9))
                    lambda_reg_scaled = self.lambda_reg * (self.s ** 2)
                    tamp_t = max((2.0 - (1.0 + self.tampering_coef) ** i), 0.0)

                    total_loss = -(self.lambda_alpha * reward_loss) \
                                 + (tamp_t * lambda_reg_scaled * weighted_eps_loss) \
                                 + (tamp_t * sig_coeff * total_emb_reg)

                    if torch.isfinite(total_loss):
                        grads = torch.autograd.grad(
                            total_loss, 
                            inputs=[un_master, pooled_un_master]
                            # retain_graph=True only until the last iteration
                            # retain_graph=(inner_idx < current_inner_steps_int - 1)
                        )
                        un_master.grad = grads[0]
                        pooled_un_master.grad = grads[1]
                        opt.step()
                    else:
                        opt.zero_grad()

            # 3. Branch / Evaluate / Select (particle filtering)
            with torch.no_grad():
                un_committed = un_master.to(self.pipe.unet.dtype)
                pooled_committed = pooled_un_master.to(self.pipe.unet.dtype)
                
                lat_in = self.pipe.scheduler.scale_model_input(torch.cat([x_t_particle]*2), t)
                t_in = t_tensor.repeat(2)
                
                emb_main = torch.cat([un_committed, cond_emb])
                emb_pool = torch.cat([pooled_committed, pooled_cond_emb])
                emb_time = torch.cat([negative_add_time_ids, add_time_ids])

                noise_pred_final = call_unet_sdxl(lat_in, t_in, emb_main, emb_pool, emb_time)
                u_final, c_final = noise_pred_final.chunk(2)
                eps_final = u_final + self.s * (c_final - u_final)

                if i < self.T - 1:
                    x_prev_in = x_t_particle.repeat(K, 1, 1, 1)
                    eps_in = eps_final.repeat(K, 1, 1, 1)
                    
                    g_step = torch.Generator(device=self.pipe.device).manual_seed(int(t))
                    x_tm1 = self.pipe.scheduler.step(eps_in, t, x_prev_in, eta=1.0, generator=g_step).prev_sample

                    tm1 = timesteps[i+1]
                    a_tm1 = alphas[int(tm1)]
                    
                    lat_tm1_in = self.pipe.scheduler.scale_model_input(torch.cat([x_tm1]*2), tm1) 
                    t_tm1 = torch.tensor([int(tm1)], device=device).repeat(2*K)
                    
                    emb_main_tm1 = torch.cat([un_committed.repeat(K,1,1), cond_emb.repeat(K,1,1)])
                    emb_pool_tm1 = torch.cat([pooled_committed.repeat(K,1), pooled_cond_emb.repeat(K,1)])
                    emb_time_tm1 = torch.cat([negative_add_time_ids.repeat(K,1), add_time_ids.repeat(K,1)])

                    noise_tm1 = call_unet_sdxl(lat_tm1_in, t_tm1, emb_main_tm1, emb_pool_tm1, emb_time_tm1)
                    u_tm1, c_tm1 = noise_tm1.chunk(2)
                    eps_tm1 = u_tm1 + self.s * (c_tm1 - u_tm1)
                    
                    x0_hat_tm1 = tweedie_x0_from_eps(x_tm1, eps_tm1, a_tm1)
                    imgs_tm1 = decode_images(self.pipe, x0_hat_tm1)
                    scores = reward_fn(imgs_tm1, [prompt]*K)
                    scores[torch.isnan(scores)] = -float('inf')

                    best_idx = torch.argmax(scores)
                    x_t_particle = x_tm1[best_idx:best_idx+1].clone()
                
                else:
                    x_t_in = x_t_particle.repeat(K, 1, 1, 1)
                    eps_in = eps_final.repeat(K, 1, 1, 1)
                    g_step = torch.Generator(device=self.pipe.device).manual_seed(int(t))
                    x_final = self.pipe.scheduler.step(eps_in, t, x_t_in, eta=1.0, generator=g_step).prev_sample
                    
                    imgs_final = decode_images(self.pipe, x_final)
                    scores = reward_fn(imgs_final, [prompt]*K)
                    best_idx = torch.argmax(scores)
                    x_t_particle = x_final[best_idx:best_idx+1].clone()

        return decode_images(self.pipe, x_t_particle)


# =========================
# Single experiment
# =========================
def run_single_experiment(
    lambda_alpha, lambda_beta, lambda_gamma, num_particles, log_dir, pipe,
    target_reward_fn, pickscore_fn, aesthetic_fn, hps_fn, imagereward_fn,
    base_tensors_batch, current_seed, current_prompt, current_negative_prompt,
    min_inner_steps, max_inner_steps, target_reward_name, tampering_coef,
    # ✨ SDXL embeddings passed explicitly
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, add_time_ids, negative_add_time_ids
):
    print(f"\n--- Starting SDXL Null-TTA (Target: {target_reward_name}) ---")

    base_img = base_tensors_batch["base_img"]
    x_T_noise_batch_all_K = base_tensors_batch["x_T_noise"].clone()

    # Baseline scores (only the target will be valid; others may be nan)
    with torch.no_grad():
        prompts = [current_prompt] * 1
        img_batch = base_img.unsqueeze(0)
        s_pick = float(pickscore_fn(img_batch, prompts).mean().item())
        s_aes = float(aesthetic_fn(img_batch, prompts).mean().item())
        s_hps = float(hps_fn(img_batch, prompts).mean().item())
        s_ir = float(imagereward_fn(img_batch, prompts).mean().item())
        print(f"Baseline: Pick={s_pick:.3f}, Aes={s_aes:.3f}, HPS={s_hps:.3f}, IR={s_ir:.3f}")

    runner = CFGOptWithBeamSearch(
        pipe, guidance_scale, num_inference_steps, lr_uncond,
        min_inner_steps, max_inner_steps, lambda_alpha, lambda_beta, lambda_gamma,
        num_particles, tampering_coef
    )

    # Optimize
    img_opt_batch = runner.optimize(
        current_prompt,
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds,
        add_time_ids, negative_add_time_ids,
        target_reward_fn,
        x_T_noise_batch_all_K
    )

    # Optimized scores
    with torch.no_grad():
        prompts = [current_prompt] * 1
        best_img = img_opt_batch[0].unsqueeze(0)
        o_pick = float(pickscore_fn(best_img, prompts).mean().item())
        o_aes = float(aesthetic_fn(best_img, prompts).mean().item())
        o_hps = float(hps_fn(best_img, prompts).mean().item())
        o_ir = float(imagereward_fn(best_img, prompts).mean().item())
        print(f"Optimized: Pick={o_pick:.3f}, Aes={o_aes:.3f}, HPS={o_hps:.3f}, IR={o_ir:.3f}")

    return (base_img, best_img.squeeze(0), s_pick, o_pick, s_aes, o_aes, s_hps, o_hps, s_ir, o_ir)


# =========================
# Main
# =========================
def main():
    global seed, max_inner_steps, min_inner_steps, num_particles, lr_uncond, tampering_coef

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_reward", type=str, default="pickscore", choices=["pickscore", "aesthetic", "hpsv2", "imagereward"])
    parser.add_argument("--lambda_alpha", type=float)
    parser.add_argument("--lambda_reg", type=float)
    parser.add_argument("--phi_variance", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--min_inner_steps", type=int)
    parser.add_argument("--max_inner_steps", type=int)
    parser.add_argument("--num_particles", type=int)
    parser.add_argument("--lr_uncond", type=float)
    parser.add_argument("--tampering_coef", type=float)
    args = parser.parse_args()

    # Update globals
    if args.seed is not None: seed = args.seed
    if args.min_inner_steps is not None: min_inner_steps = args.min_inner_steps
    if args.max_inner_steps is not None: max_inner_steps = args.max_inner_steps
    if args.num_particles is not None: num_particles = args.num_particles
    if args.lr_uncond is not None: lr_uncond = args.lr_uncond
    if args.tampering_coef is not None: tampering_coef = args.tampering_coef

    hp_triples = [(args.lambda_alpha, args.lambda_reg, args.phi_variance)] if args.lambda_alpha else DEFAULT_HYPERPARAM_TRIPLES

    base_log_dir = base_log_dir_root_template.format(
        num_particles=num_particles, min_inner_steps=min_inner_steps, max_inner_steps=max_inner_steps,
        seed=seed, target=args.target_reward
    )
    os.makedirs(base_log_dir, exist_ok=True)

    print(f"Loading SDXL: {model_id} (fp16 variant, safetensors)")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    
    # VAE to float32 for safety
    if hasattr(pipe, "vae"): pipe.vae.to(dtype=torch.float32)

    # ✨ Enable UNet checkpointing (memory saving)
    pipe.unet.enable_gradient_checkpointing()

    # ✨ Target reward only loading
    pickscore_fn = get_pickscore_fn(args.target_reward)
    aesthetic_fn = get_aesthetic_fn(args.target_reward)
    hps_fn = get_hps_fn(args.target_reward)
    imagereward_fn = get_imagereward_fn(args.target_reward)

    if args.target_reward == "pickscore": target_fn = pickscore_fn
    elif args.target_reward == "aesthetic": target_fn = aesthetic_fn
    elif args.target_reward == "hpsv2": target_fn = hps_fn
    elif args.target_reward == "imagereward": target_fn = imagereward_fn
    else: target_fn = MockReward()

    for a, b, g in hp_triples:
        log_dir = os.path.join(base_log_dir, f"A_{a}_B_{b}_G_{g}")
        os.makedirs(log_dir, exist_ok=True)
        
        with open(os.path.join(log_dir, "results.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt", "base_pick", "opt_pick", "base_aes", "opt_aes", "base_hps", "opt_hps", "base_ir", "opt_ir"])

            for idx, prompt in enumerate(prompt_list):
                print(f"\nPrompt {idx}: {prompt[:50]}...")
                
                (p_emb, _, pool_p_emb, _) = pipe.encode_prompt(
                    prompt=prompt, device=device, do_classifier_free_guidance=False
                )
                (n_p_emb, _, n_pool_p_emb, _) = pipe.encode_prompt(
                    prompt=negative_prompt, device=device, do_classifier_free_guidance=False
                )

                add_time_ids = pipe._get_add_time_ids((height, width), (0,0), (height, width), dtype=p_emb.dtype, text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).to(device)
                n_add_time_ids = add_time_ids.clone()

                gen = set_seed(seed)
                x_T = make_init_latents(pipe, height, width, num_particles, gen)

                # --- Generate baseline and save immediately ---
                base_img_path = os.path.join(log_dir, f"{idx}_base.png")
                with torch.no_grad():
                    base_latent_out_fp16 = pipe(
                        prompt=prompt, 
                        negative_prompt=negative_prompt, 
                        latents=x_T[0:1], 
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale, 
                        output_type="latent"
                    ).images 

                    base_latents_fp32 = base_latent_out_fp16.to(dtype=pipe.vae.dtype)
                    
                    latents_scaled = base_latents_fp32 / pipe.vae.config.scaling_factor
                    decoded_imgs = pipe.vae.decode(latents_scaled).sample
                    base_out = (decoded_imgs / 2 + 0.5).clamp(0, 1)
                    base_out = base_out[0]
                
                save_image_tensor(base_out, base_img_path)
                print(f"Baseline image saved to: {base_img_path}")
                # --------------------------------------------

                base_batch = {"base_img": base_out, "x_T_noise": x_T}

                res = run_single_experiment(
                    a, b, g, num_particles, log_dir, pipe,
                    target_fn, pickscore_fn, aesthetic_fn, hps_fn, imagereward_fn,
                    base_batch, seed, prompt, negative_prompt, min_inner_steps, max_inner_steps, args.target_reward, tampering_coef,
                    p_emb, n_p_emb, pool_p_emb, n_pool_p_emb, add_time_ids, n_add_time_ids
                )
                
                writer.writerow([prompt, res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9]])
                
                opt_img_path = os.path.join(log_dir, f"{idx}_opt.png")
                save_image_tensor(res[1], opt_img_path)
                print(f"Optimized image saved to: {opt_img_path}")

if __name__ == "__main__":
    main()