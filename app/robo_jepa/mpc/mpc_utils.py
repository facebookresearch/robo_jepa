# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import logging
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from app.robo_jepa.inference.mem_debug import log_rss

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# a simple cos similarity
def cos(a, b):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return (a * b).sum(-1)


def l1(a, b):
    return -torch.mean(torch.abs(a - b), dim=-1)


def cem(
    world_model,
    wm_state,
    goal_features,
    rollout=3,
    cem_steps=5,
    action_dim=7,
    action_repeat=1,
    momentum_mean=0.0,
    momentum_std=0.25,
    momentum_mean_gripper=0.0,
    momentum_std_gripper=0.25,
    actions_suffix=None,
    samples=100,
    cutoff=0.95,
    verbose=False,
    use_gripper=False,
    maxnorm=0.05,
    view_weights=None,
    robot_init_std=0.05,
    gripper_init_std=0.10,
    maxgripper=0.10,
    objective=l1,
    translation_only=True,
    num_batches=1,
    prev_actions=None,
    prev_stds=None,
    world_size=1,
    rank=0,
    step_func=None,
    device="cuda",
    use_cache=True,
    global_step=0,
    total_steps=100,
):
    """Plan an action sequence with the cross-entropy method (CEM) in the world model's latent space.

    Each CEM iteration samples candidate action sequences from a per-step Gaussian, rolls every candidate out through
    the world model from the current context, scores the imagined features against the goal features, keeps the
    top-scoring sequences (elites), and refits the Gaussian to them with momentum. Sampling is data-parallel: each rank
    draws ``num_batches * samples`` candidates, and rewards and actions are all-gathered before elite selection.

    Actions are 7-D: translation (3), rotation (3, euler) and gripper (1), in the units the world model expects.
    Per step, a candidate is sampled as follows:

    * translation: a random direction and radius (``r ~ U(0, 1)``, two angles ``~ U(0, 2π)``), scaled by the std and
      shifted by the mean, then projected onto the L2 ball of radius ``maxnorm``;
    * rotation: ``N(mean, std)``, set to 0 when ``translation_only``;
    * gripper: ``N(mean, std)`` clipped to ``[-maxgripper, maxgripper]`` when ``use_gripper``, else 0.

    Args:
        world_model: the ``WorldModel`` (used for ``world_model.step`` when ``step_func`` is None).
        wm_state: ``WorldModelState`` holding the current context (batch 1); it is repeated ``samples`` times.
        goal_features: goal encoder features, ``[1, 1, V, H, W, C]``, scored only against the last imagined step;
            or a dict ``{absolute_step: features}`` of (sub-)goals, scored at every rollout step whose absolute index
            ``global_step + h + 1`` is a key (all views flattened into one score).
        rollout: planning horizon in WM steps; clipped to ``total_steps - global_step``.
        cem_steps: number of CEM iterations.
        action_dim: action width (6 robot dims + 1 gripper).
        action_repeat: each sampled action is applied in this many WM steps of ``action / action_repeat``.
        momentum_mean, momentum_std: momentum for the robot (first 6 dims) mean / std update:
            ``new = elite_stat * (1 - m) + old * m``.
        momentum_mean_gripper, momentum_std_gripper: the same for the gripper dim.
        actions_suffix: optional ground-truth actions, only used for verbose logging.
        samples: candidates per batch (and the batch size of each world-model rollout).
        cutoff: elites to keep; ``> 1`` is an absolute count, otherwise the top ``1 - cutoff`` fraction of all
            gathered candidates (at least 3).
        verbose: log progress and statistics on rank 0.
        use_gripper: sample the gripper dim (otherwise it is 0).
        maxnorm: maximum L2 norm of the per-step translation.
        view_weights: per-view weights of the return used for elite selection (default: uniform).
        robot_init_std: initial std of the robot dims.
        gripper_init_std: initial std of the gripper dim.
        maxgripper: bound on the sampled gripper action.
        objective: ``objective(goal, predicted)`` -> reward per sample (higher is better); default: negative mean L1.
        translation_only: zero the rotation dims.
        num_batches: sequential sample batches per rank and CEM iteration.
        prev_actions: ``[n, action_dim]`` plan from the previous env step, used as the initial mean (padded with
            zeros to ``rollout``); zeros if None.
        prev_stds: accepted but unused; the std always restarts from the initial values.
        world_size, rank: distributed setup; with ``world_size > 1`` rewards and actions are all-gathered.
        step_func: rollout function with the ``WorldModel.step`` signature; defaults to ``world_model.step``.
        device: device for the world-model rollout.
        use_cache: roll out through the world model's KV cache.
        global_step: index of the current env step within the episode.
        total_steps: episode length in env steps (bounds the horizon and indexes sub-goals).

    Returns:
        (rewards, means, stds), one entry per CEM iteration:

        * rewards: ``[V]`` tensor, the per-view return (sum of rewards over the horizon) averaged over the elites;
        * means: ``[rollout, action_dim]`` refit mean; ``means[-1]`` is the plan to execute;
        * stds: ``[rollout, action_dim]`` refit std.

        The mean/std of a rollout step is only updated when that step or a later one received a non-zero reward.
    """
    if step_func is None:
        step_func = world_model.step
    rollout = min(rollout, total_steps - global_step)
    if verbose and rank == 0:
        logger.info(f"Planning for step {global_step} with rollout {rollout}")
    if prev_actions is None:
        mean = torch.zeros((rollout, action_dim))
    else:
        mean = prev_actions
        if len(prev_actions) < rollout:
            new_actions = torch.zeros((rollout - len(prev_actions), action_dim))
            mean = torch.cat([mean, new_actions])
    if verbose and rank == 0:
        logger.debug(f"rank {rank}/{world_size}:: initial_mean: {mean.shape}\n {mean}")
    # The CEM std always restarts from the initial std (``prev_stds`` is accepted but not used).
    std = torch.cat(
        [
            torch.ones((rollout, action_dim - 1)) * robot_init_std,
            torch.ones((rollout, 1)) * gripper_init_std,
        ],
        dim=1,
    )
    if translation_only:
        std[:, 3:6] = 0.0
    if verbose and rank == 0:
        logger.debug(f"rank {rank}/{world_size}:: initial_std: {std.shape}\n {std}")
    wm_state = wm_state.repeat(samples)
    log_rss(f"cem:after_repeat[gs={global_step}]")
    _, _, V, H, W, C = wm_state.features.shape
    mean_list = []
    std_list = []
    rewards = []
    for step in tqdm(range(cem_steps), disable=not (verbose and rank == 0)):
        action_batches = []
        reward_batches = []
        for batch_no in range(num_batches):
            action_stats = []
            rewards_sequence = torch.zeros((samples, rollout, V))
            t = time.time()
            history = wm_state.to(device)
            for h in range(rollout):
                if verbose and rank == 0:
                    logger.debug(f"rank {rank}/{world_size}:: running step {h}")
                r = np.random.uniform(
                    size=samples,
                )
                th1 = np.random.uniform(size=samples) * 2 * np.pi
                th2 = np.random.uniform(size=samples) * 2 * np.pi
                x = r * np.cos(th1) * np.cos(th2)
                y = r * np.cos(th1) * np.sin(th2)
                z = r * np.sin(th1)
                action_samples = np.stack([x, y, z], axis=1)
                rotation_samples = np.random.normal(size=(samples, 3))
                action_samples = torch.from_numpy(action_samples).float()
                rotation_samples = torch.from_numpy(rotation_samples).float()
                action_samples = torch.cat([action_samples, rotation_samples], dim=1)
                action_samples = action_samples * std[h][:6] + mean[h][:6]
                if translation_only:
                    action_samples[:, 3:6] = 0.0
                action_norms = action_samples[:, :3].norm(p=2, dim=1, keepdim=True)
                action_samples[:, :3] = torch.where(
                    action_norms < maxnorm,
                    action_samples[:, :3],
                    action_samples[:, :3] / action_norms * maxnorm,
                )
                if use_gripper:
                    # TODO: try sigmoid for gripper
                    gripper_actions = torch.randn((samples, 1)) * std[h][6:] + mean[h][6:]
                    # logger.info(f'clipping gripper in [{-maxgripper}, {maxgripper}]')
                    gripper_actions.clip_(min=-maxgripper, max=maxgripper)
                else:
                    gripper_actions = torch.zeros((samples, 1))
                action_samples = torch.cat([action_samples, gripper_actions], dim=1)
                # action_samples = torch.randn(samples, 7, device=mean.device) * std[h] + mean[h]
                # action_samples[:, :3] = torch.clip(action_samples[:, :3], min=-maxnorm, max=maxnorm)
                # action_samples[:, -1:] = torch.clip(action_samples[:, -1:], min=-0.75, max=0.75)
                new_action = action_samples[:, None]
                action_stats.append(new_action.cpu())
                for k in range(action_repeat):
                    state, history = step_func(
                        new_action / action_repeat,
                        history,
                        no_grad=True,
                        device=device,
                        use_cache=use_cache,
                        batch=100,
                    )
                if isinstance(goal_features, dict):
                    if (h + global_step + 1) in goal_features:
                        sims = objective(
                            goal_features[h + global_step + 1].reshape(1, -1),
                            state.reshape(samples, -1).to(device),
                        ).cpu()
                        rewards_sequence[:, h] = sims
            d = time.time() - t
            if verbose and rank == 0:
                logger.debug(f"rank {rank}/{world_size}:: rollout time : {d:.2f}sec; per step : {d / rollout:.3f}sec")
            # OOM mitigation: drop the per-batch CEM rollout history wm_state
            # (carries the per-rollout AOT KV caches and per-program input
            # buffers) as soon as the batch is done. ``state`` is still
            # needed below for the per-batch ``objective(...)`` call, so
            # leave it bound. ``history`` is the heavy one (its KV cache
            # tensors are 100s of MB on GPU even after ``.cpu()`` on the
            # outer wm_state).
            del history
            gc.collect()
            action_stats = torch.cat(action_stats, dim=1)
            action_batches.append(action_stats.cpu())
            reward_batches.append(rewards_sequence)
            if verbose and rank == 0:
                logger.debug(f"rank {rank}/{world_size}:: action stats: {action_stats.sum(0).sum(0)}")
            if not isinstance(goal_features, dict):
                sims = objective(goal_features.flatten(3, 5), state.flatten(3, 5).to(device))
                rewards_sequence[:, h : h + 1] = sims.cpu()
        reward_batches = torch.cat(reward_batches, dim=0).to(device)
        action_stats = torch.cat(action_batches, dim=0).to(device)
        if world_size != 1:
            dist.barrier()
            reward_placeholders = [torch.zeros_like(reward_batches) for _ in range(world_size)]
            dist.all_gather(reward_placeholders, reward_batches)
            reward_batches = torch.cat(reward_placeholders)
            dist.barrier()
            action_stats0 = action_stats.to(device)
            action_placeholders = [torch.zeros_like(action_stats0) for _ in range(world_size)]
            dist.all_gather(action_placeholders, action_stats0)
            action_stats = torch.cat(action_placeholders).cpu()
            dist.barrier()
        else:
            reward_batches = reward_batches.cpu()
            action_stats = action_stats.cpu()
        # dist.barrier()
        # TODO: this is a very naive way of estimating returns
        returns = reward_batches.sum(1)
        sims = returns.clone()
        if verbose and rank == 0:
            logger.info(f"rank {rank}/{world_size}:: step {step}")
            view_sims = "; ".join(
                f"view {l}:: mean: {sims[:, l].mean():.4f}, std: {sims[:, l].std():.4f}"
                for l in range(sims.shape[1])  # noqa: E741
            )
            logger.info(f"rank {rank}/{world_size}:: mean sims: {sims.mean():4f}, std: {sims.std():.4f}; {view_sims}")
        if cutoff > 1:
            cutoff_n = int(cutoff)
        else:
            cutoff_n = max(int(num_batches * samples * world_size * (1 - cutoff)), 3)
        if view_weights is None:
            view_weights = [1.0 / returns.shape[1] for _ in range(returns.shape[1])]
        for j, w in enumerate(view_weights):
            returns[:, j] *= w
        top_sims = returns.sum(1).cpu().topk(cutoff_n, largest=True)
        if verbose and rank == 0:
            top_view_sims = sims[top_sims.indices]
            view_sims = "; ".join(
                f"view {l}:: mean: {top_view_sims[:, l].mean():.4f}, std: {top_view_sims[:, l].std():.4f}"
                for l in range(top_view_sims.shape[1])  # noqa: E741
            )
            logger.info(
                f"rank {rank}/{world_size}:: top sims mean: {top_sims.values.mean():.4f} std: {top_sims.values.std():.4f}; {view_sims}"  # noqa: E501
            )
        indices = top_sims.indices
        if verbose and rank == 0:
            logger.info(f"rank {rank}/{world_size}:: stats shape: {action_stats.shape}, indices: {indices.shape}")
        selected_actions = action_stats[indices]
        new_mean = selected_actions.mean(0)
        new_std = selected_actions.std(0)
        # if rank == 0:
        #     import pdb; pdb.set_trace()
        # if world_size != 1:
        #     dist.barrier()
        # reward_mask.shape == (rollout,) and position k is True iff there is nonzero reward at pos >=k
        reward_mask = (reward_batches.sum(2) != 0).any(0).flip(0).long().cumsum(0).flip(0).bool()
        # if rank == 0:
        #    import pdb; pdb.set_trace()
        # if world_size != 1:
        #    dist.barrier()
        mean_update_robot = new_mean[:, :6] * (1 - momentum_mean) + mean[:, :6] * momentum_mean
        std_update_robot = new_std[:, :6] * (1 - momentum_std) + std[:, :6] * momentum_std
        mean_update_gripper = new_mean[:, 6:] * (1 - momentum_mean_gripper) + mean[:, 6:] * momentum_mean_gripper
        std_update_gripper = new_std[:, 6:] * (1 - momentum_std_gripper) + std[:, 6:] * momentum_std_gripper
        mean_update = torch.cat([mean_update_robot, mean_update_gripper], dim=1)
        std_update = torch.cat([std_update_robot, std_update_gripper], dim=1)

        mean = torch.where(reward_mask.unsqueeze(1).cpu(), mean_update, mean)
        std = torch.where(reward_mask.unsqueeze(1).cpu(), std_update, std)
        if verbose and rank == 0:
            logger.info(
                f"rank {rank}/{world_size}:: new mean:\n{mean} "
            )  # ({mean[:, :3] * action_repeat} after repeat)')
            logger.info(
                f"rank {rank}/{world_size}:: new std:\n{std} "
            )  # ({mean[:, :3] * action_repeat} after repeat)')
            if actions_suffix is not None:
                logger.debug(f"rank {rank}/{world_size}:: optimal actions:\n{actions_suffix[0][:, :3]}")
                logger.debug(f"rank {rank}/{world_size}:: optimal delta pose:\n{actions_suffix[0][:, :3].sum(0)}")
            logger.debug(f"rank {rank}/{world_size}:: new std:\n{std}")
            if actions_suffix is not None:
                logger.debug(
                    f"rank {rank}/{world_size}:: l1 distance from the current mean to the optimal actions: {torch.abs(actions_suffix[0] - mean[0]).sum():.4f}"  # noqa: E501
                )
                logger.debug(
                    f"rank {rank}/{world_size}:: l1 (xyz) distance from the current mean to the optimal actions: {torch.abs(actions_suffix[0] - mean[0])[..., :3].sum():.4f}"  # noqa: E501
                )
                d_pose_gt = actions_suffix[0].sum(0)[:3]
                d_pose_gt = d_pose_gt / d_pose_gt.norm()
                d_pose_mpc = mean[0][:3]
                d_pose_mpc = d_pose_mpc / d_pose_mpc.norm()
                logger.debug(
                    f"rank {rank}/{world_size}:: cos (xyz) similarity from the current mean to the optimal actions: {(d_pose_gt * d_pose_mpc).sum():.4f}"  # noqa: E501
                )
                logger.debug(
                    f"rank {rank}/{world_size}:: pose delta diff: {torch.abs(actions_suffix[0].sum(0)[:3] - mean[0][:3]).sum():.4f}"  # noqa: E501
                )
        mean_list.append(mean.clone())
        std_list.append(std.clone())
        # rewards.append(top_sims.values.mean())
        rewards.append(sims[top_sims.indices].mean(0))
        log_rss(f"cem:end_iter[gs={global_step},step={step}]")
    log_rss(f"cem:return[gs={global_step}]")
    return rewards, mean_list, std_list
