# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from .noise_schedule import make_beta_schedule
from .respace import SpacedDiffusion, space_timesteps


def create_diffusion(
    timestep_respacing,
    noise_schedule="cosine_simple_diffusion",
    shift=1.0,
    clip_min=1e-9,
    zero_terminal_snr=True,
    use_kl=False,
    sigma_small=True,
    predict_xstart=False,
    predict_v=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
):
    # betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    betas = make_beta_schedule(
        noise_schedule, shift=shift, clip_min=clip_min, zero_terminal_snr=zero_terminal_snr, timesteps=diffusion_steps
    )
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    mean_type = None
    assert int(predict_xstart) + int(predict_v) <= 1
    if predict_xstart:
        mean_type = gd.ModelMeanType.START_X
    elif predict_v:
        mean_type = gd.ModelMeanType.VELOCITY
    else:
        mean_type = gd.ModelMeanType.EPSILON
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=mean_type,
        model_var_type=(
            (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        # rescale_timesteps=rescale_timesteps,
    )
