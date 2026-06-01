import os
import random
from typing import List, Tuple, Deque, Dict
import statistics
import csv
from collections import deque
import math  # For torch.pi
import argparse  # Argument parsing
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline, DDIMScheduler
from diffusers import DDPMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from tqdm.auto import tqdm

# =========================
# Config
# =========================
device = "cuda"
dtype_pipe = torch.float16
height, width = 512, 512

# Default value (overridden by CLI args)
seed = 42

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

negative_prompt = ""

num_samples = 3
num_particles = 3
num_inference_steps = 100
guidance_scale = 7.5

min_inner_steps = 5
max_inner_steps = 25
lr_uncond = 1e-2
tampering_coef = 0.008  # Used as (1 + tampering_coef)**i

DEFAULT_HYPERPARAM_TRIPLES = [
    (100, 0.002, 0.01)
]

# Converted into a template string to format seed / max_step / target
base_log_dir_root_template = (
    "logs/final_particle_{num_particles}_tampering_steps_"
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
    latents = (latents / pipe.vae.config.scaling_factor).to(pipe.vae.dtype)
    imgs = pipe.vae.decode(latents).sample
    imgs = ((imgs / 2) + 0.5).clamp(0, 1)
    return imgs.float()


def decode_images_grad(pipe: DiffusionPipeline, latents: torch.Tensor) -> torch.Tensor:
    latents = (latents / pipe.vae.config.scaling_factor).to(pipe.vae.dtype)
    imgs = pipe.vae.decode(latents).sample
    imgs = ((imgs / 2) + 0.5).clamp(0, 1)
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


# --- reward loaders ---

class MockRewardNaN:
    def __call__(self, images, prompts):
        if isinstance(prompts, str):
            prompts = [prompts] * images.shape[0]
        return torch.tensor([float('nan')] * images.shape[0], device=images.device)

    def eval(self):
        pass


def get_reward_fn(target_name, force_load=False):
    """PickScore (default optimization target)"""
    if not force_load and target_name != "pickscore":
        return MockRewardNaN()

    try:
        import das.rewards as rewards
        reward_model = rewards.PickScore(device=device)
        print("Using PickScore reward.")
        return reward_model
    except Exception as e:
        print(f"Could not import das.rewards or load PickScore: {e}")

        class MockReward:
            def __call__(self, images, prompts):
                if isinstance(prompts, str):
                    prompts = [prompts] * images.shape[0]
                return torch.randn(images.shape[0], device=images.device) * 2.5

            def eval(self):
                pass

        print("Using a mock PickScore reward function (for optimization/logging).")
        return MockReward()


def get_aesthetic_fn(target_name, force_load=False):
    if not force_load and target_name != "aesthetic":
        return MockRewardNaN()

    try:
        import das.rewards as rewards
        reward_model = rewards.aesthetic_score(device=device)
        print("Using aesthetic_score reward (for logging).")
        return reward_model
    except Exception as e:
        print(f"Could not import das.rewards or load aesthetic_score: {e}")

        class MockAesthetic:
            def __call__(self, images, prompts):
                if isinstance(prompts, str):
                    prompts = [prompts] * images.shape[0]
                return torch.tensor([float('nan')] * images.shape[0], device=images.device)

            def eval(self):
                pass

        print("Using a mock aesthetic function (returns NaN).")
        return MockAesthetic()


def get_hps_fn(target_name, force_load=False):
    if not force_load and target_name != "hpsv2":
        return MockRewardNaN()

    try:
        import das.rewards as rewards
        reward_model = rewards.hps_score(device=device)
        print("Using hps_score reward (for logging).")
        return reward_model
    except Exception as e:
        print(f"Could not import das.rewards or load hps_score: {e}")

        class MockHPS:
            def __call__(self, images, prompts):
                if isinstance(prompts, str):
                    prompts = [prompts] * images.shape[0]
                return torch.randn(images.shape[0], device=images.device) * 0.5

            def eval(self):
                pass

        print("Using a mock hps_score function (for logging).")
        return MockHPS()


def get_imagereward_fn(target_name, force_load=False):
    if not force_load and target_name != "imagereward":
        return MockRewardNaN()

    try:
        import das.rewards as rewards
        reward_model = rewards.ImageReward(device=device)
        print("Using ImageReward reward (for logging).")
        return reward_model
    except Exception as e:
        print(f"Could not import das.rewards or load ImageReward: {e}")

        class MockImageReward:
            def __call__(self, images, prompts):
                if isinstance(prompts, str):
                    prompts = [prompts] * images.shape[0]
                return torch.tensor([float('nan')] * images.shape[0], device=images.device)

            def eval(self):
                pass

        print("Using a mock ImageReward function (returns NaN).")
        return MockImageReward()


def save_image_tensor(img: torch.Tensor, path: str):
    if img.ndim == 4:
        img = img[0]
    arr = (img.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    Image.fromarray(arr).save(path)
    print("Saved:", path)


def make_init_latents(pipe: DiffusionPipeline, h: int, w: int, batch_size: int, gen: torch.Generator):
    lat_h, lat_w = h // 8, w // 8
    return torch.randn((batch_size, 4, lat_h, lat_w), generator=gen, device=pipe.device, dtype=pipe.unet.dtype)


# =========================
# Core optimizer
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

        for p in self.pipe.unet.parameters():
            p.requires_grad_(False)
        if hasattr(self.pipe, "vae"):
            for p in self.pipe.vae.parameters():
                p.requires_grad_(False)
        if hasattr(self.pipe, "text_encoder"):
            for p in self.pipe.text_encoder.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def _encode_text_pair(self, prompt: str, negative_prompt: str):
        tok = self.pipe.tokenizer
        max_len = tok.model_max_length
        uncond = tok(
            [negative_prompt or ""],
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        ).to(self.pipe.device)
        un_emb = self.pipe.text_encoder(uncond.input_ids)[0]
        cond = tok(
            [prompt],
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        ).to(self.pipe.device)
        cond_emb = self.pipe.text_encoder(cond.input_ids)[0]
        return un_emb.to(self.pipe.unet.dtype), cond_emb.to(self.pipe.unet.dtype)

    def optimize(
        self,
        prompt: str,
        negative_prompt: str,
        reward_fn,  # target reward fn
        x_T_init_batch: torch.Tensor,
    ) -> torch.Tensor:

        K = self.K_samples
        if x_T_init_batch.shape[0] != K:
            print(f"Warning: x_T_init_batch size ({x_T_init_batch.shape[0]}) != K ({K}). Using first particle.")
            x_T_init_batch = x_T_init_batch[0:1].repeat(K, 1, 1, 1)

        un_orig, cond_emb = self._encode_text_pair(prompt, negative_prompt)
        un_master = un_orig.detach().float().clone().requires_grad_(True)

        s = self.pipe.scheduler
        timesteps = s.timesteps
        alphas = s.alphas_cumprod.to(self.pipe.device)
        betas = s.betas.to(self.pipe.device)

        do_classifier_free_guidance = self.s > 1.0

        # initial selection
        with torch.no_grad():
            print(f"v15: Performing initial selection from {K} noise particles at t={timesteps[0]}...")
            try:
                t_start = timesteps[0]
                t_tensor_start = torch.tensor([int(t_start)], device=x_T_init_batch.device, dtype=torch.long)
                a_t_start = alphas[int(t_start)]

                un_orig_batch = un_orig.repeat(K, 1, 1)
                cond_emb_batch = cond_emb.repeat(K, 1, 1)

                latent_model_input_start = torch.cat([x_T_init_batch] * 2) if do_classifier_free_guidance else x_T_init_batch
                t_tensor_start_unet = (
                    t_tensor_start.repeat(K * 2) if do_classifier_free_guidance else t_tensor_start.repeat(K)
                )

                latent_model_input_start = self.pipe.scheduler.scale_model_input(latent_model_input_start, t_start)
                combined_embeds_start = (
                    torch.cat([un_orig_batch, cond_emb_batch]) if do_classifier_free_guidance else cond_emb_batch
                )

                noise_pred_start = self.pipe.unet(
                    latent_model_input_start,
                    t_tensor_start_unet,
                    encoder_hidden_states=combined_embeds_start,
                    return_dict=False,
                )[0]

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
                print(f"v15: Initial particle {best_idx_start} selected (Score: {scores_start[best_idx_start]:.4f}).")

            except Exception as e:
                print(f"ERROR during initial selection: {e}. Defaulting to particle 0.")
                import traceback
                traceback.print_exc()
                x_t_particle = x_T_init_batch[0:1].clone()

        for i, t in enumerate(
            tqdm(
                timesteps,
                desc="Optimizing with Particle Filter (v15 + VI_LOSS + NO_HIST_REG)",
                leave=False,
            )
        ):
            t_int = int(t)
            t_tensor = torch.tensor([t_int], device=x_t_particle.device, dtype=torch.long)
            a_t_bar = alphas[t_int]

            # baseline eps_u at this step
            with torch.no_grad():
                latent_model_input_orig = torch.cat([x_t_particle] * 2) if do_classifier_free_guidance else x_t_particle
                latent_model_input_orig = self.pipe.scheduler.scale_model_input(latent_model_input_orig, t)

                t_tensor_orig_unet = t_tensor.repeat(2) if do_classifier_free_guidance else t_tensor

                combined_embeds_orig = torch.cat([un_orig, cond_emb]) if do_classifier_free_guidance else cond_emb
                noise_pred_orig = self.pipe.unet(
                    latent_model_input_orig,
                    t_tensor_orig_unet,
                    encoder_hidden_states=combined_embeds_orig,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond_orig, _ = noise_pred_orig.chunk(2)
                    eps_u_orig_particle = noise_pred_uncond_orig
                else:
                    eps_u_orig_particle = noise_pred_orig

            # tampering-based inner steps
            # uses (1 + tampering_coef)**i
            tampering_progress = min(1.0, (1.0 + self.tampering_coef) ** i - 1.0)
            current_inner_steps = self.min_inner_steps + (self.max_inner_steps - self.min_inner_steps) * tampering_progress
            current_inner_steps_int = int(round(current_inner_steps))

            current_lr = self.lr_uncond
            opt = torch.optim.Adam([un_master], lr=current_lr)

            last_reward_value = None  # buffer for printing per-step reward

            for inner_idx in range(current_inner_steps_int):
                opt.zero_grad(set_to_none=True)

                un_cur = un_master.to(self.pipe.unet.dtype)

                # === current step only (NO_HIST_REG) ===
                latent_model_input = self.pipe.scheduler.scale_model_input(x_t_particle, t_int)
                t_tensor_input = t_tensor

                un_cur_batch = un_cur
                cond_emb_batch = cond_emb

                latent_model_input_reward_batch = (
                    torch.cat([latent_model_input] * 2) if do_classifier_free_guidance else latent_model_input
                )
                combined_embeds_reward_batch = (
                    torch.cat([un_cur_batch, cond_emb_batch]) if do_classifier_free_guidance else cond_emb_batch
                )
                t_tensor_unet_input = (
                    t_tensor_input.repeat(2) if do_classifier_free_guidance else t_tensor_input
                )

                noise_pred_reward_batch = self.pipe.unet(
                    latent_model_input_reward_batch,
                    t_tensor_unet_input,
                    encoder_hidden_states=combined_embeds_reward_batch,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond_batch, noise_pred_text_batch = noise_pred_reward_batch.chunk(2)
                    eps_cfg_target_batch = noise_pred_uncond_batch + self.s * (
                        noise_pred_text_batch - noise_pred_uncond_batch
                    )
                else:
                    noise_pred_uncond_batch = noise_pred_reward_batch
                    eps_cfg_target_batch = noise_pred_reward_batch

                try:
                    a_t_bar_view = a_t_bar.view(-1, 1, 1, 1)

                    x0_hat_target_batch = tweedie_x0_from_eps(x_t_particle, eps_cfg_target_batch, a_t_bar_view)
                    imgs_batch_reward = decode_images_grad(self.pipe, x0_hat_target_batch)

                    # Optimize using the target reward
                    R_batch = reward_fn(imgs_batch_reward, [prompt] * 1)
                    R_batch[torch.isnan(R_batch)] = 0.0
                    last_reward_value = float(R_batch[0].item())
                except Exception as e:
                    print(f"ERROR decoding or getting reward in inner loop at step {t_int}: {e}")
                    R_batch = torch.zeros(1, device=x_t_particle.device)

                eps_u_target_batch = noise_pred_uncond_batch

                batch_betas = betas[t_int]
                batch_a_t_bars_reg = alphas[t_int]
                batch_a_t_individuals = (1.0 - batch_betas).clamp(min=1e-9)
                batch_one_minus_a_t_bars = (1.0 - batch_a_t_bars_reg).clamp(min=1e-9)

                batch_denominators = 2.0 * batch_a_t_individuals * batch_one_minus_a_t_bars

                lambda_t_coeff_batch = torch.where(
                    batch_denominators > 1e-9,
                    batch_betas / batch_denominators,
                    torch.tensor(1e9, device=betas.device, dtype=betas.dtype),
                )

                per_sample_sq_error = F.mse_loss(
                    eps_u_target_batch, eps_u_orig_particle.detach(), reduction='none'
                )
                per_sample_mse_sum = per_sample_sq_error.sum(dim=(1, 2, 3))

                weighted_eps_losses = lambda_t_coeff_batch * per_sample_mse_sum

                sig_coeff = self.lambda_reg * (1.0 / (2.0 * self.phi_variance + 1e-9))
                lambda_reward = self.lambda_alpha
                lambda_reg_scaled = self.lambda_reg * (self.s ** 2)

                emb_loss = F.mse_loss(un_master, un_orig.float(), reduction='sum')

                i_t = i
                tamp_t = max((2.0 - (1.0 + self.tampering_coef) ** i_t), 0.0)

                reward_t = R_batch[0]
                eps_loss_t = weighted_eps_losses[0]

                loss_t_obj = (lambda_reward * reward_t) - \
                             (tamp_t * lambda_reg_scaled * eps_loss_t) - \
                             (tamp_t * sig_coeff * emb_loss)

                loss = -loss_t_obj

                if torch.isfinite(loss):
                    loss.backward()
                    opt.step()
                else:
                    opt.zero_grad()

            if last_reward_value is not None:
                print(
                    f"[Step {i + 1}/{self.T}] Approx target reward: {last_reward_value:.4f}"
                )

            # branch / evaluate / select
            with torch.no_grad():
                un_committed = un_master.to(self.pipe.unet.dtype)

                latent_model_input_final = (
                    torch.cat([x_t_particle] * 2) if do_classifier_free_guidance else x_t_particle
                )
                latent_model_input_final = self.pipe.scheduler.scale_model_input(latent_model_input_final, t)

                t_tensor_final_unet = t_tensor.repeat(2) if do_classifier_free_guidance else t_tensor

                combined_embeds_final = (
                    torch.cat([un_committed, cond_emb]) if do_classifier_free_guidance else cond_emb
                )

                noise_pred_final = self.pipe.unet(
                    latent_model_input_final,
                    t_tensor_final_unet,
                    encoder_hidden_states=combined_embeds_final,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond_final, noise_pred_text_final = noise_pred_final.chunk(2)
                    eps_step_single = noise_pred_uncond_final + self.s * (
                        noise_pred_text_final - noise_pred_uncond_final
                    )
                else:
                    eps_step_single = noise_pred_final

                if i < self.T - 1:
                    x_t_to_step = x_t_particle.repeat(self.K_samples, 1, 1, 1)
                    eps_to_step = eps_step_single.repeat(self.K_samples, 1, 1, 1)

                    g_step = torch.Generator(device=self.pipe.device).manual_seed(int(t))

                    x_tm1_samples = self.pipe.scheduler.step(
                        eps_to_step,
                        t,
                        x_t_to_step,
                        eta=1.0,
                        generator=g_step,
                    ).prev_sample

                    try:
                        tm1 = timesteps[i + 1]
                        t_tensor_tm1 = torch.tensor(
                            [int(tm1)], device=x_tm1_samples.device, dtype=torch.long
                        )
                        a_tm1 = alphas[int(tm1)]

                        un_committed_batch = un_committed.repeat(self.K_samples, 1, 1)
                        cond_emb_batch = cond_emb.repeat(self.K_samples, 1, 1)

                        latent_model_input_tm1 = (
                            torch.cat([x_tm1_samples] * 2) if do_classifier_free_guidance else x_tm1_samples
                        )
                        t_tensor_tm1_unet = (
                            t_tensor_tm1.repeat(self.K_samples * 2)
                            if do_classifier_free_guidance
                            else t_tensor_tm1.repeat(self.K_samples)
                        )

                        latent_model_input_tm1 = self.pipe.scheduler.scale_model_input(
                            latent_model_input_tm1, tm1
                        )
                        combined_embeds_tm1 = (
                            torch.cat([un_committed_batch, cond_emb_batch])
                            if do_classifier_free_guidance
                            else cond_emb_batch
                        )

                        noise_pred_tm1 = self.pipe.unet(
                            latent_model_input_tm1,
                            t_tensor_tm1_unet,
                            encoder_hidden_states=combined_embeds_tm1,
                            return_dict=False,
                        )[0]

                        if do_classifier_free_guidance:
                            noise_pred_uncond_tm1, noise_pred_text_tm1 = noise_pred_tm1.chunk(2)
                            eps_cfg_tm1 = noise_pred_uncond_tm1 + self.s * (
                                noise_pred_text_tm1 - noise_pred_uncond_tm1
                            )
                        else:
                            eps_cfg_tm1 = noise_pred_tm1

                        x0_hat_tm1_samples = tweedie_x0_from_eps(x_tm1_samples, eps_cfg_tm1, a_tm1)
                        imgs_tm1 = decode_images(self.pipe, x0_hat_tm1_samples)

                        scores_tm1 = reward_fn(imgs_tm1, [prompt] * self.K_samples)
                        scores_tm1[torch.isnan(scores_tm1)] = -float('inf')

                    except Exception as e:
                        print(f"ERROR during evaluation step at t={t_int}: {e}")
                        import traceback
                        traceback.print_exc()
                        scores_tm1 = torch.zeros(self.K_samples, device=x_t_particle.device)
                        scores_tm1[0] = 1.0

                    _, best_idx = torch.max(scores_tm1, dim=0)
                    x_t_particle = x_tm1_samples[best_idx:best_idx + 1].clone()

                else:
                    x_t_to_step = x_t_particle.repeat(self.K_samples, 1, 1, 1)
                    eps_to_step = eps_step_single.repeat(self.K_samples, 1, 1, 1)

                    g_step = torch.Generator(device=self.pipe.device).manual_seed(int(t))
                    x_tm1_candidates = self.pipe.scheduler.step(
                        eps_to_step,
                        t,
                        x_t_to_step,
                        eta=1.0,
                        generator=g_step,
                    ).prev_sample

                    imgs_final = decode_images(self.pipe, x_tm1_candidates)
                    scores = reward_fn(imgs_final, [prompt] * self.K_samples)
                    scores[torch.isnan(scores)] = -float('inf')
                    best_idx = torch.argmax(scores)
                    x_t_particle = x_tm1_candidates[best_idx:best_idx + 1].clone()

        with torch.no_grad():
            img_opt_batch = decode_images(self.pipe, x_t_particle)
        return img_opt_batch


# =========================
# Single experiment
# =========================
def run_single_experiment(
    lambda_alpha,
    lambda_beta,
    lambda_gamma,
    num_particles,
    log_dir,
    pipe,
    target_reward_fn,
    pickscore_fn,
    aesthetic_fn,
    hps_fn,
    imagereward_fn,
    base_tensors_batch,
    current_seed: int,
    current_prompt: str,
    current_negative_prompt: str,
    min_inner_steps: int,
    max_inner_steps: int,
    target_reward_name: str,
    tampering_coef: float,
):
    print(
        f"\n--- Starting Particle Filter Experiment [v15 + VI Loss + NO_HIST_REG]"
        f" (Opt Target: {target_reward_name})"
        f": A={lambda_alpha}, L_Reg(B)={lambda_beta}, Var(G)={lambda_gamma},"
        f" K={num_particles}, S={current_seed} ---"
    )

    base_img = base_tensors_batch["base_img"]
    x_T_noise_batch_all_K = base_tensors_batch["x_T_noise"].clone()

    # baseline scores (all four metrics)
    with torch.no_grad():
        try:
            prompt_list_base = [current_prompt] * base_img.unsqueeze(0).shape[0]
            score_base_pick = float(pickscore_fn(base_img.unsqueeze(0), prompt_list_base).mean().item())
            score_base_aesthetic = float(aesthetic_fn(base_img.unsqueeze(0), prompt_list_base).mean().item())
            score_base_hps = float(hps_fn(base_img.unsqueeze(0), prompt_list_base).mean().item())
            score_base_imagereward = float(
                imagereward_fn(base_img.unsqueeze(0), prompt_list_base).mean().item()
            )
            print(f"Using pre-generated baseline (seed {current_seed}).")
            print(
                f"  Baseline Scores: Pick={score_base_pick:.4f},"
                f" Aes={score_base_aesthetic:.4f},"
                f" HPS={score_base_hps:.4f},"
                f" IR={score_base_imagereward:.4f}"
            )
        except Exception as e:
            print(f"ERROR calculating baseline scores: {e}")
            score_base_pick = float("nan")
            score_base_aesthetic = float("nan")
            score_base_hps = float("nan")
            score_base_imagereward = float("nan")

    runner = CFGOptWithBeamSearch(
        pipe,
        guidance_scale,
        num_inference_steps,
        lr_uncond,
        min_inner_steps=min_inner_steps,
        max_inner_steps=max_inner_steps,
        lambda_alpha=lambda_alpha,
        lambda_beta=lambda_beta,
        lambda_gamma=lambda_gamma,
        num_beams=num_particles,
        tampering_coef=tampering_coef,
    )

    for fn in [target_reward_fn, pickscore_fn, aesthetic_fn, hps_fn, imagereward_fn]:
        if hasattr(fn, "eval"):
            fn.eval()

    print(
        f"--- Starting optimization (K={num_particles} initial particles,"
        f" K={num_particles} samples/step, NO_HIST_REG)"
        f" (Opt Target: {target_reward_name}) ---"
    )

    # optimized scores
    score_opt_pick = -float("inf")
    score_opt_aesthetic = float("nan")
    score_opt_hps = float("nan")
    score_opt_imagereward = float("nan")

    try:
        print(f"--- [!!] Optimizing using {target_reward_name} as the target reward function [!!] ---")
        img_opt_batch = runner.optimize(
            current_prompt,
            current_negative_prompt,
            target_reward_fn,
            x_T_init_batch=x_T_noise_batch_all_K,
        )

        with torch.no_grad():
            prompt_list_opt = [current_prompt] * img_opt_batch.shape[0]

            scores_opt_pick_batch = pickscore_fn(img_opt_batch, prompt_list_opt)
            score_opt_pick = float(scores_opt_pick_batch[0].item())
            if np.isnan(score_opt_pick):
                score_opt_pick = -float("inf")

            scores_opt_aesthetic_batch = aesthetic_fn(img_opt_batch, prompt_list_opt)
            score_opt_aesthetic = float(scores_opt_aesthetic_batch[0].item())
            if np.isnan(score_opt_aesthetic):
                score_opt_aesthetic = float("nan")

            scores_opt_hps_batch = hps_fn(img_opt_batch, prompt_list_opt)
            score_opt_hps = float(scores_opt_hps_batch[0].item())
            if np.isnan(score_opt_hps):
                score_opt_hps = float("nan")

            scores_opt_imagereward_batch = imagereward_fn(img_opt_batch, prompt_list_opt)
            score_opt_imagereward = float(scores_opt_imagereward_batch[0].item())
            if np.isnan(score_opt_imagereward):
                score_opt_imagereward = float("nan")

        best_img = img_opt_batch[0]
        # Image saving is handled in main where file naming is standardized
        print(
            f"Final optimized scores: Pick={score_opt_pick:.4f},"
            f" Aes={score_opt_aesthetic:.4f},"
            f" HPS={score_opt_hps:.4f},"
            f" IR={score_opt_imagereward:.4f}"
        )

    except Exception as e:
        print(f"ERROR during particle filter optimization: {e}")
        import traceback
        traceback.print_exc()
        score_opt_pick = -float("inf")
        score_opt_aesthetic = float("nan")
        score_opt_hps = float("nan")
        score_opt_imagereward = float("nan")
        best_img = base_img  # fallback

    final_best_pick_score = score_opt_pick if score_opt_pick > -float("inf") else float("nan")

    return (
        base_img,
        best_img,
        score_base_pick,
        final_best_pick_score,
        score_base_aesthetic,
        score_opt_aesthetic,
        score_base_hps,
        score_opt_hps,
        score_base_imagereward,
        score_opt_imagereward,
    )


# =========================
# Main
# =========================
def main():
    global seed, max_inner_steps, min_inner_steps, num_particles, num_inference_steps, lr_uncond, tampering_coef

    parser = argparse.ArgumentParser(
        description="Run Particle Filter Optimization (No Historical Regulation)"
    )

    # Select target reward
    parser.add_argument(
        "--target_reward",
        type=str,
        default="pickscore",
        choices=["pickscore", "aesthetic", "hpsv2"],
        help="Which reward to optimize: pickscore, aesthetic, hpsv2",
    )

    # Hyperparameters (existing + additional args)
    parser.add_argument("--lambda_alpha", type=float, help="Single value for lambda_alpha (Alpha)")
    parser.add_argument("--lambda_reg", type=float, help="Single value for lambda_reg (Beta)")
    parser.add_argument("--phi_variance", type=float, help="Single value for phi_variance (Gamma)")
    parser.add_argument("--seed", type=int, help="Random seed for generation")

    parser.add_argument(
        "--min_inner_steps",
        type=int,
        help="Min inner optimization steps per diffusion step (default: 5)",
    )
    parser.add_argument(
        "--max_inner_steps",
        type=int,
        help="Max inner optimization steps per diffusion step (default: 25)",
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        help="Number of particles for particle filter (default: 3)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        help="Number of diffusion inference steps (default: 100)",
    )
    parser.add_argument(
        "--lr_uncond",
        type=float,
        help="Learning rate for null-text optimization (default: 1e-2)",
    )
    parser.add_argument(
        "--tampering_coef",
        type=float,
        help="Tampering coefficient used in (1 + tampering_coef)**i (default: 0.008)",
    )
    parser.add_argument("--eval_all_rewards", action="store_true")

    args = parser.parse_args()

    # Override defaults
    if args.seed is not None:
        seed = args.seed
    if args.min_inner_steps is not None:
        min_inner_steps = args.min_inner_steps
    if args.max_inner_steps is not None:
        max_inner_steps = args.max_inner_steps
    if args.num_particles is not None:
        num_particles = args.num_particles
    if args.num_inference_steps is not None:
        num_inference_steps = args.num_inference_steps
    if args.lr_uncond is not None:
        lr_uncond = args.lr_uncond
    if args.tampering_coef is not None:
        tampering_coef = args.tampering_coef

    # Set hyperparameter triples (retain original search mode)
    if args.lambda_alpha is not None and args.lambda_reg is not None and args.phi_variance is not None:
        hp_triples = [(args.lambda_alpha, args.lambda_reg, args.phi_variance)]
        print(
            f"--- Running in SINGLE mode for HP: "
            f"A={args.lambda_alpha}, L_Reg(B)={args.lambda_reg}, Var(G)={args.phi_variance} ---"
        )
    else:
        hp_triples = DEFAULT_HYPERPARAM_TRIPLES
        print(f"--- Running in SEARCH mode for {len(hp_triples)} HP triples ---")

    target_reward_name = {
        "pickscore": "PickScore",
        "aesthetic": "AestheticScore",
        "hpsv2": "HPSv2",
    }[args.target_reward]

    # Create log directory name reflecting seed / steps / target
    base_log_dir_root = base_log_dir_root_template.format(
        num_particles=num_particles,
        min_inner_steps=min_inner_steps,
        max_inner_steps=max_inner_steps,
        seed=seed,
        target=args.target_reward,
    )
    os.makedirs(base_log_dir_root, exist_ok=True)

    print("--- Loading models and setting up common resources (ONCE) ---")

    print(
        "Loading model from user-specified cache: "
        "stable-diffusion-v1-5/stable-diffusion-v1-5"
    )
    pipe = DiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=dtype_pipe,
        safety_checker=None,
    ).to(device)
    if hasattr(pipe, "vae"):
        pipe.vae.to(dtype=torch.float32)
    if hasattr(pipe, "text_encoder"):
        pipe.text_encoder.to(dtype=torch.float32)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(num_inference_steps, device=pipe.device)

    try:
        ddpm_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.betas = ddpm_scheduler.betas
        print("Successfully copied .betas from DDPMScheduler to DDIMScheduler.")
    except Exception as e:
        print(f"Could not copy betas, ensure scheduler config is compatible. Error: {e}")
        if not hasattr(pipe.scheduler, "betas"):
            raise ValueError(
                "Scheduler does not have .betas attribute, "
                "cannot proceed with v5.6/v15 formula."
            )

    pickscore_fn = get_reward_fn(args.target_reward, force_load=args.eval_all_rewards)
    aesthetic_fn = get_aesthetic_fn(args.target_reward, force_load=args.eval_all_rewards)
    hps_fn = get_hps_fn(args.target_reward, force_load=args.eval_all_rewards)
    imagereward_fn = get_imagereward_fn(args.target_reward, force_load=args.eval_all_rewards)

    # Select target reward
    if args.target_reward == "pickscore":
        target_reward_fn = pickscore_fn
    elif args.target_reward == "aesthetic":
        target_reward_fn = aesthetic_fn
    elif args.target_reward == "hpsv2":
        target_reward_fn = hps_fn
    else:
        target_reward_fn = pickscore_fn  # fallback

    print(f"--- [!!] OPTIMIZATION TARGET SET TO {target_reward_name} [!!] ---")

    all_avg_results = []

    print("\n" + "=" * 60)
    print(f"========= RUNNING FOR SINGLE SEED: {seed} =========")
    print(
        f"========= Averaging over {len(prompt_list)} prompts"
        f" (v15 Tampering Steps + VI_LOSS + Multi-Score + NO_HIST_REG)"
        f" (Opt Target: {target_reward_name}) ========="
    )
    print("=" * 60)

    for a_val, b_val, g_val in hp_triples:
        current_pair_prompt_results = []

        hyperparam_log_dir = os.path.join(
            base_log_dir_root,
            f"alpha_{a_val}_LReg-B_{b_val}_Var-G_{g_val}",
        )
        os.makedirs(hyperparam_log_dir, exist_ok=True)

        print(
            f"\n{'=' * 20} Testing HP Triple:"
            f" Alpha={a_val}, L_Reg(B)={b_val}, Var(G)={g_val} "
            f"{'=' * 20}"
        )

        # Store all prompts + averages in a single CSV
        csv_path = os.path.join(
            hyperparam_log_dir,
            f"results_{args.target_reward}.csv",
        )

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "prompt_idx",
                    "prompt",
                    "baseline_pick",
                    "optimized_pick",
                    "baseline_aes",
                    "optimized_aes",
                    "baseline_hps",
                    "optimized_hps",
                    "baseline_ir",
                    "optimized_ir",
                ]
            )

            for prompt_idx, current_prompt in enumerate(prompt_list):
                print(f"\n--- Prompt {prompt_idx + 1}/{len(prompt_list)} (Seed: {seed}) ---")
                print(f"--- P: {current_prompt[:70]}...")

                g_baseline = set_seed(seed)
                x_T_noise_batch = make_init_latents(
                    pipe, height, width, num_particles, g_baseline
                )

                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=dtype_pipe
                ):
                    out_base = pipe(
                        prompt=current_prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        eta=1.0,
                        latents=x_T_noise_batch[0:1].clone(),
                        output_type="pt",
                    )
                selected_base_img = out_base.images[0]

                base_tensors_batch = {
                    "base_img": selected_base_img,
                    "x_T_noise": x_T_noise_batch,
                }

                (
                    base_img,
                    opt_img,
                    base_score_pick,
                    best_opt_pick_score,
                    base_score_aes,
                    best_opt_aes_score,
                    base_score_hps,
                    best_opt_hps_score,
                    base_score_ir,
                    best_opt_ir_score,
                ) = run_single_experiment(
                    lambda_alpha=a_val,
                    lambda_beta=b_val,
                    lambda_gamma=g_val,
                    num_particles=num_particles,
                    log_dir=hyperparam_log_dir,
                    pipe=pipe,
                    target_reward_fn=target_reward_fn,
                    pickscore_fn=pickscore_fn,
                    aesthetic_fn=aesthetic_fn,
                    hps_fn=hps_fn,
                    imagereward_fn=imagereward_fn,
                    base_tensors_batch=base_tensors_batch,
                    current_seed=seed,
                    current_prompt=current_prompt,
                    current_negative_prompt=negative_prompt,
                    min_inner_steps=min_inner_steps,
                    max_inner_steps=max_inner_steps,
                    target_reward_name=target_reward_name,
                    tampering_coef=tampering_coef,
                )

                # Image saving: only base/opt in the same folder
                base_img_path = os.path.join(
                    hyperparam_log_dir,
                    f"base_prompt_{prompt_idx:03d}_target-{args.target_reward}.png",
                )
                opt_img_path = os.path.join(
                    hyperparam_log_dir,
                    f"opt_prompt_{prompt_idx:03d}_target-{args.target_reward}.png",
                )
                save_image_tensor(base_img, base_img_path)
                save_image_tensor(opt_img, opt_img_path)

                writer.writerow(
                    [
                        prompt_idx,
                        current_prompt,
                        base_score_pick,
                        best_opt_pick_score,
                        base_score_aes,
                        best_opt_aes_score,
                        base_score_hps,
                        best_opt_hps_score,
                        base_score_ir,
                        best_opt_ir_score,
                    ]
                )

                current_pair_prompt_results.append(
                    {
                        "base_pick": base_score_pick,
                        "opt_pick": best_opt_pick_score,
                        "base_aes": base_score_aes,
                        "opt_aes": best_opt_aes_score,
                        "base_hps": base_score_hps,
                        "opt_hps": best_opt_hps_score,
                        "base_ir": base_score_ir,
                        "opt_ir": best_opt_ir_score,
                        "prompt": current_prompt,
                    }
                )

            # Compute averages & append to the last row of the CSV
            def safe_mean(vals):
                vals = [v for v in vals if not np.isnan(v)]
                return statistics.mean(vals) if vals else float("nan")

            avg_base_pick = safe_mean(
                [r["base_pick"] for r in current_pair_prompt_results]
            )
            avg_opt_pick = safe_mean(
                [r["opt_pick"] for r in current_pair_prompt_results]
            )
            avg_base_aes = safe_mean(
                [r["base_aes"] for r in current_pair_prompt_results]
            )
            avg_opt_aes = safe_mean(
                [r["opt_aes"] for r in current_pair_prompt_results]
            )
            avg_base_hps = safe_mean(
                [r["base_hps"] for r in current_pair_prompt_results]
            )
            avg_opt_hps = safe_mean(
                [r["opt_hps"] for r in current_pair_prompt_results]
            )
            avg_base_ir = safe_mean(
                [r["base_ir"] for r in current_pair_prompt_results]
            )
            avg_opt_ir = safe_mean(
                [r["opt_ir"] for r in current_pair_prompt_results]
            )

            # Decide which metric to use for "improvement": based on target reward
            if args.target_reward == "pickscore":
                avg_improvement_target = avg_opt_pick - avg_base_pick
                target_base_avg = avg_base_pick
                target_opt_avg = avg_opt_pick
            elif args.target_reward == "aesthetic":
                avg_improvement_target = avg_opt_aes - avg_base_aes
                target_base_avg = avg_base_aes
                target_opt_avg = avg_opt_aes
            elif args.target_reward == "hpsv2":
                avg_improvement_target = avg_opt_hps - avg_base_hps
                target_base_avg = avg_base_hps
                target_opt_avg = avg_opt_hps
            else:
                avg_improvement_target = avg_opt_pick - avg_base_pick
                target_base_avg = avg_base_pick
                target_opt_avg = avg_opt_pick

            writer.writerow(
                [
                    "avg",
                    "",
                    avg_base_pick,
                    avg_opt_pick,
                    avg_base_aes,
                    avg_opt_aes,
                    avg_base_hps,
                    avg_opt_hps,
                    avg_base_ir,
                    avg_opt_ir,
                ]
            )

        print("\n" + "*" * 60)
        print(
            f"*** HP Triple Average (A={a_val}, L_Reg(B)={b_val}, Var(G)={g_val}) ***"
        )
        print(
            f"*** ({target_reward_name}) Base: {target_base_avg:.4f}, "
            f"Opt: {target_opt_avg:.4f}, Improv: {avg_improvement_target:.4f} "
            f"(Optimization Target & Sorting Metric)"
        )
        print(f"*** (PickScore)  Base: {avg_base_pick:.4f}, Opt: {avg_opt_pick:.4f}")
        print(f"*** (HPSv2)      Base: {avg_base_hps:.4f}, Opt: {avg_opt_hps:.4f}")
        print(f"*** (Aesthetic)  Base: {avg_base_aes:.4f}, Opt: {avg_opt_aes:.4f}")
        print(f"*** (ImageRwrd)  Base: {avg_base_ir:.4f}, Opt: {avg_opt_ir:.4f}")
        print(
            f"*** (Based on {len(current_pair_prompt_results)} / {len(prompt_list)} prompts)"
        )
        print("*" * 60 + "\n")

        all_avg_results.append(
            {
                "lambda_alpha": a_val,
                "lambda_reg (from beta)": b_val,
                "phi_variance (from gamma)": g_val,
                "num_particles": num_particles,
                "min_inner_steps": min_inner_steps,
                "max_inner_steps": max_inner_steps,
                "method": f"particle_filter_{num_particles}_v15_tampering_VI_LOSS_NO_HIST_REG_avg (Opt: {target_reward_name})",
                "avg_baseline_pickscore": avg_base_pick,
                "avg_optimized_pickscore": avg_opt_pick,
                "avg_baseline_aesthetic_score": avg_base_aes,
                "avg_optimized_aesthetic_score": avg_opt_aes,
                "avg_baseline_hps_score": avg_base_hps,
                "avg_optimized_hps_score": avg_opt_hps,
                "avg_baseline_imagereward_score": avg_base_ir,
                "avg_optimized_imagereward_score": avg_opt_ir,
                "avg_target_baseline": target_base_avg,
                "avg_target_optimized": target_opt_avg,
                "avg_improvement_target": avg_improvement_target,
                "num_valid_prompts": len(current_pair_prompt_results),
            }
        )

    print("\n\n" + "=" * 50)
    print(
        "           HYPERPARAMETER SEARCH COMPLETE "
        f"(v15 - Tampering Steps + VI_LOSS + Multi-Score + NO_HIST_REG)"
        f" (Opt Target: {target_reward_name})"
    )
    print("=" * 50)
    print(
        f"Overall Average Results (Method: Particle Filter "
        f"K={num_particles} v15_tampering_VI_LOSS_NO_HIST_REG "
        f"(Opt Target: {target_reward_name}), Steps={min_inner_steps}-{max_inner_steps}):"
    )

    # Sort by target reward improvement
    sorted_results = sorted(
        [r for r in all_avg_results if not np.isnan(r.get("avg_improvement_target", float("nan")))],
        key=lambda x: x.get("avg_improvement_target", -float("inf")),
        reverse=True,
    )

    print("\n--- Results sorted by Average Target Reward Improvement (Best first) ---")
    for result in sorted_results:
        imp_str = f"{result['avg_improvement_target']:.4f}"
        opt_pick_str = f"{result['avg_optimized_pickscore']:.4f}"
        opt_aes_str = f"{result['avg_optimized_aesthetic_score']:.4f}"
        opt_hps_str = f"{result['avg_optimized_hps_score']:.4f}"
        opt_ir_str = f"{result['avg_optimized_imagereward_score']:.4f}"

        print(
            f"A: {result['lambda_alpha']:<6.1f}, "
            f"L_Reg(B): {result['lambda_reg (from beta)']:<6.4f}, "
            f"Var(G): {result['phi_variance (from gamma)']:<5.2f} "
            f"-> Improv(Target): {imp_str:<10}"
        )
        print(
            f"     -> Opt Scores (Pick: {opt_pick_str:<10} | "
            f"Aes: {opt_aes_str:<10} | HPS: {opt_hps_str:<10} | IR: {opt_ir_str:<10})"
        )

    summary_path = os.path.join(
        base_log_dir_root,
        f"hyperparam_summary_PROMPT_AVG_v15_TAMP_STEPS_VI_LOSS_MULTI_SCORE_NO_HIST_REG_target-{args.target_reward}.csv",
    )
    file_exists = os.path.isfile(summary_path)

    with open(summary_path, "a", newline="") as csvfile:
        fieldnames = [
            "lambda_alpha",
            "lambda_reg (from beta)",
            "phi_variance (from gamma)",
            "num_particles",
            "min_inner_steps",
            "max_inner_steps",
            "method",
            "avg_baseline_pickscore",
            "avg_optimized_pickscore",
            "avg_baseline_aesthetic_score",
            "avg_optimized_aesthetic_score",
            "avg_baseline_hps_score",
            "avg_optimized_hps_score",
            "avg_baseline_imagereward_score",
            "avg_optimized_imagereward_score",
            "avg_target_baseline",
            "avg_target_optimized",
            "avg_improvement_target",
            "num_valid_prompts",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for row in sorted_results:
            row_to_write = {}
            for key in fieldnames:
                if key in row:
                    value = row[key]
                    if isinstance(value, float) and np.isnan(value):
                        row_to_write[key] = "nan"
                    else:
                        row_to_write[key] = value
                else:
                    row_to_write[key] = "N/A"
            writer.writerow(row_to_write)

    print(
        f"\nComprehensive average summary (v15, VI_LOSS, multi-score, NO_HIST_REG,"
        f" target={target_reward_name}) saved/appended to: {summary_path}"
    )
    print("-> Use this file and per-HP results_*.csv to pick Pareto points.")


if __name__ == "__main__":
    main()