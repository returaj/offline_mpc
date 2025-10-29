import os
import os.path as osp
import random
import re
import sys
import time
from collections import deque
from copy import deepcopy

import dsrl.infos as dsrl_infos
import dsrl.offline_safety_gymnasium  # type: ignore
import gymnasium as gym
import numpy as np
import torch
import torch.distributions as td
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR

from dsrl_model.utils.bufffer import SafeDiceBuffer
from dsrl_model.utils.dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_neg_and_union_data_2,
)
from dsrl_model.utils.logger import EpochLogger
from dsrl_model.utils.models import (
    DwbcDiscriminator,
    SafeDiceTanhMixtureActor,
    gradient_panelty,
    minmax_discriminator_loss,
)
from dsrl_model.utils.save_video_with_value import save_video
from dsrl_model.utils.utils import ActionRepeater, get_params_norm, single_agent_args

EP = 1e-6
EP2 = 1e-3

LOG_PI_NORM_MAX = 10
LOG_PI_NORM_MIN = -60

default_cfg = {
    "log_freq": int(1e4),
    "save_freq": int(2e4),
    "eval_episode_freq": 1,
    "cost_update_freq": 100,
    "hidden_size": 256,
    "max_grad_norm": 10.0,
    "gamma": 0.99,
    "action_repeat": 1,
    "actor_lr": 1e-5,
    "critic_lr": 1e-5,
    "dwbc_nu": 0.5,
    "total_iteration": int(1e6),
    "weight_decay_cost": 0.01,
}

trajectory_cfg = {
    "density": 1.0,
    "target_cost": 25.0,
    # ((low_cost, low_reward), (high_cost, low_reward), (medium_cost, high_reward))
    "inpaint_ranges": ((0.0, 1.0, 0.0, 0.5),),
    "num_negative_trajectories": 50,
    "num_union_trajectories": -1,
    "percentage_validation_trajectories": 0.2,
}


@torch.no_grad
def evaluate_bc_policy(eval_env, bc_policy, device, save_video=False):
    eval_done = False
    eval_obs, _ = eval_env.reset()
    # eval_obs = (eval_obs - mu_obs) / (std_obs + EP)
    eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32, device=device).unsqueeze(
        0
    )
    eval_reward, eval_cost, eval_len = (
        0.0,
        0.0,
        0.0,
    )
    ep_frames, ep_costs = [], []
    while not eval_done:
        act = bc_policy.action(eval_obs)
        next_obs, reward, terminated, truncated, info = eval_env.step(
            act[0].detach().squeeze().cpu().numpy()
        )
        cost = info["cost"]
        # next_obs = (next_obs - mu_obs) / (std_obs + EP)
        next_obs = torch.as_tensor(
            next_obs, dtype=torch.float32, device=device
        ).unsqueeze(0)
        eval_obs = next_obs
        eval_reward += reward
        eval_cost += cost
        eval_len += 1
        eval_done = terminated or truncated
        if save_video:
            ep_frames.append(eval_env.render())
            ep_costs.append(
                {
                    "C(s,a)": cost,
                    "tC(s,a)": eval_cost,
                }
            )
    return eval_reward, eval_cost, eval_len, ep_frames, ep_costs


def cost_loss_fn(
    critic_model,
    actor,
    target_neg_obs,
    target_neg_act,
    target_union_obs,
    target_union_act,
    config,
):
    nu = config["dwbc_nu"]

    with torch.no_grad():
        log_pi_neg = actor.true_get_logprob(target_neg_obs, target_neg_act)
        log_pi_union = actor.true_get_logprob(target_union_obs, target_union_act)
        log_pi_neg_clip = torch.clip(log_pi_neg, LOG_PI_NORM_MIN, LOG_PI_NORM_MAX)
        log_pi_union_clip = torch.clip(log_pi_union, LOG_PI_NORM_MIN, LOG_PI_NORM_MAX)
        log_pi_neg_norm = (log_pi_neg_clip - LOG_PI_NORM_MIN) / (
            LOG_PI_NORM_MAX - LOG_PI_NORM_MIN
        )
        log_pi_neg_norm = log_pi_neg_norm.unsqueeze(dim=1)
        log_pi_union_norm = (log_pi_union_clip - LOG_PI_NORM_MIN) / (
            LOG_PI_NORM_MAX - LOG_PI_NORM_MIN
        )
        log_pi_union_norm = log_pi_union_norm.unsqueeze(dim=1)

    cost_neg = critic_model(target_neg_obs, target_neg_act, log_pi_neg_norm)
    cost_union = critic_model(target_union_obs, target_union_act, log_pi_union_norm)

    loss_neg = -torch.log(cost_neg)
    loss_union = -torch.log(1 - cost_union) / nu + torch.log(1 - cost_neg)
    return torch.mean(loss_neg + loss_union)


def bc_loss_fn(
    critic_model,
    actor,
    target_union_obs,
    target_union_act,
):
    with torch.no_grad():
        log_pi_union = actor.true_get_logprob(target_union_obs, target_union_act)
        log_pi_union_clip = torch.clip(log_pi_union, LOG_PI_NORM_MIN, LOG_PI_NORM_MAX)
        log_pi_union_norm = (log_pi_union_clip - LOG_PI_NORM_MIN) / (
            LOG_PI_NORM_MAX - LOG_PI_NORM_MIN
        )
        log_pi_union_norm = log_pi_union_norm.unsqueeze(dim=1)
        cost_union = critic_model(target_union_obs, target_union_act, log_pi_union_norm)

    weight = 1 - cost_union
    pred_act, *_ = actor(target_union_obs)
    recon_loss = F.mse_loss(pred_act, target_union_act, reduction="none").sum(dim=1)
    loss = weight * recon_loss
    # log_prob = -actor.true_get_logprob(target_union_obs, target_union_act)
    # loss = weight * log_prob
    return torch.mean(loss)


def main(args, cfg_env=None):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cpu.deterministic = True
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device(f"{args.device}:{args.device_id}")

    trajectory_cfg["num_negative_trajectories"] = args.num_non_preferred
    trajectory_cfg["num_union_trajectories"] = args.num_union
    trajectory_cfg["non_pref_noise"] = args.non_pref_noise

    config = {**default_cfg, **trajectory_cfg}
    config["dwbc_nu"] = args.dwbc_nu or config["dwbc_nu"]

    # evaluation environment
    eval_env = gym.make(args.task)
    if args.save_video:
        eval_env.render_parameters.mode = "rgb_array"
        eval_env.render_parameters.camera_name = "track"
    eval_env.set_target_cost(config["target_cost"])
    eval_env = ActionRepeater(eval_env, num_repeats=config["action_repeat"])
    eval_env.reset(seed=args.seed)

    # set training steps
    batch_size = args.batch_size or config.get("batch_size")

    # set model
    obs_space, act_space = eval_env.observation_space, eval_env.action_space
    # critic model
    critic_model = DwbcDiscriminator(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
    ).to(device)
    critic_model_optimizer = torch.optim.AdamW(
        critic_model.parameters(),
        lr=config["critic_lr"],
        weight_decay=config["weight_decay_cost"],
    )
    # actor / policy model
    actor = SafeDiceTanhMixtureActor(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_size=config["hidden_size"],
    ).to(device)
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=config["actor_lr"],
        weight_decay=config["weight_decay_cost"],
    )
    actor_scheduler = LinearLR(
        actor_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=config["total_iteration"],
    )

    # data
    agent_task = re.search(r"Offline(.*?)Gymnasium-v[0-9]", args.task).group(1)
    ep_len = dsrl_infos.DEFAULT_MAX_EPISODE_STEPS[agent_task]
    data = get_dataset_in_d4rl_format(
        eval_env, trajectory_cfg, args.task, ep_len, config["action_repeat"]
    )
    neg_data, union_data = get_neg_and_union_data_2(data, trajectory_cfg)
    # neg_data, union_data, mu_obs, std_obs = get_normalized_data(neg_data, union_data)
    neg_observations = torch.as_tensor(
        neg_data["observations"], dtype=torch.float32, device=device
    )
    neg_actions = torch.as_tensor(
        neg_data["actions"], dtype=torch.float32, device=device
    )
    neg_dones = neg_data["timeouts"] | neg_data["terminals"]
    union_observations = torch.as_tensor(
        union_data["observations"], dtype=torch.float32, device=device
    )
    union_actions = torch.as_tensor(
        union_data["actions"], dtype=torch.float32, device=device
    )
    union_dones = union_data["timeouts"] | union_data["terminals"]

    ep_len = ep_len // config["action_repeat"] + (ep_len % config["action_repeat"] > 0)
    assert (
        neg_observations.shape[1] == ep_len
    ), f"{neg_observations.shape[1]} episode length is different from {ep_len}"

    buffer = SafeDiceBuffer(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        neg_data_size=np.prod(neg_observations.shape[:-1]),
        union_data_size=np.prod(union_observations.shape[:-1]),
        batch_size=batch_size,
        device=device,
        ep_len=ep_len,
    )
    for obs, act, done in zip(neg_observations, neg_actions, neg_dones):
        buffer.add(obs, act, done, is_negative=True)
    for obs, act, done in zip(union_observations, union_actions, union_dones):
        buffer.add(obs, act, done, is_negative=False)

    # set logger
    dict_args = config
    dict_args.update((k, v) for k, v in vars(args).items() if v is not None)
    logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
    )
    logger.save_config(dict_args)

    # train critic and actor model
    eval_rew_deque = deque(maxlen=config["eval_episode_freq"])
    eval_cost_deque = deque(maxlen=config["eval_episode_freq"])
    eval_norm_rew_deque = deque(maxlen=config["eval_episode_freq"])
    eval_norm_cost_deque = deque(maxlen=config["eval_episode_freq"])
    eval_len_deque = deque(maxlen=config["eval_episode_freq"])

    logger.log("Start with critic and actor model training.")
    steps = 0
    while steps < config["total_iteration"]:
        # for epoch in range(num_epochs):

        for (
            _,
            target_neg_obs,
            target_neg_act,
            _,
            target_union_obs,
            target_union_act,
            _,
        ) in buffer.sample():

            cost_loss = cost_loss_fn(
                critic_model=critic_model,
                actor=actor,
                target_neg_obs=target_neg_obs,
                target_neg_act=target_neg_act,
                target_union_obs=target_union_obs,
                target_union_act=target_union_act,
                config=config,
            )
            if steps % config["cost_update_freq"] == 0:
                critic_model_optimizer.zero_grad()
                cost_loss.backward()
                clip_grad_norm_(critic_model.parameters(), config["max_grad_norm"])
                critic_model_optimizer.step()

            pi_loss = bc_loss_fn(
                critic_model=critic_model,
                actor=actor,
                target_union_obs=target_union_obs,
                target_union_act=target_union_act,
            )
            actor_optimizer.zero_grad()
            pi_loss.backward()
            clip_grad_norm_(actor.parameters(), config["max_grad_norm"])
            actor_optimizer.step()
            actor_scheduler.step()

            logger.logged = False

            steps += 1

            if (steps % config["log_freq"] == 0) and (not logger.logged):
                # evaluate episodes
                eval_episodes = config["eval_episode_freq"]
                if args.use_eval:
                    eval_start_time = time.time()
                    for id in range(eval_episodes):
                        eval_reward, eval_cost, eval_len, *_ = evaluate_bc_policy(
                            eval_env=eval_env,
                            bc_policy=actor,
                            device=device,
                        )
                        norm_reward, norm_cost = eval_env.get_normalized_score(
                            eval_reward, eval_cost
                        )
                        eval_norm_rew_deque.append(norm_reward)
                        eval_norm_cost_deque.append(norm_cost)
                        eval_rew_deque.append(eval_reward)
                        eval_cost_deque.append(eval_cost)
                        eval_len_deque.append(eval_len)
                    logger.store(
                        **{
                            "Metrics/EvalEpRet": np.mean(eval_rew_deque),
                            "Metrics/EvalEpCost": np.mean(eval_cost_deque),
                            "Metrics/EvalEpNormRet": np.mean(eval_norm_rew_deque),
                            "Metrics/EvalEpNormCost": np.mean(eval_norm_cost_deque),
                            "Metrics/EvalEpLen": np.mean(eval_len_deque),
                        }
                    )
                    eval_end_time = time.time()

                    logger.log_tabular("Metrics/EvalEpRet")
                    logger.log_tabular("Metrics/EvalEpCost")
                    logger.log_tabular("Metrics/EvalEpNormRet")
                    logger.log_tabular("Metrics/EvalEpNormCost")
                    logger.log_tabular("Metrics/EvalEpLen")
                    logger.log_tabular("Time/Eval", eval_end_time - eval_start_time)

                logger.log_tabular("Train/Steps", steps)
                logger.log_tabular("Loss/Loss_bc_policy", pi_loss.mean().item())
                logger.log_tabular("Loss/Loss_critic", cost_loss.mean().item())
                logger.log_tabular(
                    "Norm/Params/bc_policy",
                    get_params_norm(actor.parameters(), grads=False),
                )
                logger.log_tabular(
                    "Norm/Params/critic_model",
                    get_params_norm(critic_model.parameters(), grads=False),
                )
                logger.log_tabular(
                    "Norm/Grad/bc_policy",
                    get_params_norm(actor.parameters(), grads=True),
                )
                logger.log_tabular(
                    "Norm/Grad/critic_model",
                    get_params_norm(critic_model.parameters(), grads=True),
                )
                logger.dump_tabular()

            if steps % config["save_freq"] == 0:
                logger.torch_save(
                    itr=steps,
                    torch_saver_elements=actor,
                    prefix="bc_policy",
                )
                logger.torch_save(
                    itr=steps,
                    torch_saver_elements=critic_model,
                    prefix="critic_model",
                )

            if steps >= config["total_iteration"]:
                break

    logger.torch_save(itr=steps, torch_saver_elements=actor, prefix="bc_policy")
    logger.torch_save(
        itr=steps, torch_saver_elements=critic_model, prefix="critic_model"
    )
    logger.close()


if __name__ == "__main__":
    args, cfg_env = single_agent_args()
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, algo, relpath)
    if not args.write_terminal:
        terminal_log_name = "terminal.log"
        error_log_name = "error.log"
        terminal_log_name = f"seed{args.seed}_{terminal_log_name}"
        error_log_name = f"seed{args.seed}_{error_log_name}"
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        with open(
            os.path.join(
                f"{args.log_dir}",
                terminal_log_name,
            ),
            "w",
            encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(
                    f"{args.log_dir}",
                    error_log_name,
                ),
                "w",
                encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                main(args, cfg_env)
    else:
        main(args, cfg_env)
