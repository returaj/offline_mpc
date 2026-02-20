import dsrl.infos as dsrl_infos
import matplotlib.pyplot as plt
import numpy as np

EP = 1e-7


def get_post_processed_dataset(env, data, config, task):
    density = config["density"]
    cbins, rbins = 10, 50
    max_npb, min_npb = 30, 2
    if density < 1.0:
        density_cfg = dsrl_infos.DENSITY_CFG[task + "_density" + str(density)]
        cbins, rbins = density_cfg["cbins"], density_cfg["rbins"]
        max_npb, min_npb = density_cfg["max_npb"], density_cfg["min_npb"]

    data = env.pre_process_data(
        data_dict=data,
        inpaint_ranges=config["inpaint_ranges"],
        density=density,
        cbins=cbins,
        rbins=rbins,
        max_npb=max_npb,
        min_npb=min_npb,
    )
    return data


def to_d4rl_format(data, ep_len):
    dones_idx = np.where((data["terminals"] == 1) | (data["timeouts"] == 1))[0]
    d4rl_data = {k: [] for k in data.keys()}
    for i in range(dones_idx.shape[0]):
        start = 0 if i == 0 else dones_idx[i - 1] + 1
        end = dones_idx[i] + 1
        for k, v in data.items():
            val = v[start:end]
            if ep_len != (end - start):
                repeat_len = ep_len - (end - start)
                other_dim = (1,) * (len(val.shape) - 1)
                last_val = 0.0 if (k == "rewards" or k == "costs") else val[-1]
                repeat_val = np.tile(last_val, (repeat_len, *other_dim))
                val = np.concatenate([val, repeat_val])
            d4rl_data[k].append(val)
    return {k: np.array(v) for k, v in d4rl_data.items()}


def fold_sa_pair(data: np.array, num_folds):
    assert num_folds > 0, "number of folds cannot be less than 1."
    folded_data = []
    for traj in data:
        for idx in range(num_folds):
            t = idx
            folded_traj = []
            while t < traj.shape[0]:
                v = traj[t]
                t += 1
                for _ in range(1, num_folds):
                    t += 1
                    if (t < traj.shape[0]) and (np.isscalar(traj[t])):
                        v += traj[t]
                folded_traj.append(v)
            folded_data.append(folded_traj)
    return np.array(folded_data)


def get_dataset_in_d4rl_format(env, config, task, ep_len, num_folds=1):
    data = env.get_dataset()
    data = get_post_processed_dataset(env, data, config, task)

    d4rl_data = to_d4rl_format(data, ep_len)
    keys = ["observations", "actions", "rewards", "costs", "terminals", "timeouts"]
    return {k: fold_sa_pair(d4rl_data[k], num_folds) for k in keys}


def get_reward_pos_neg_and_union_data(d4rl_data, config):
    traj_reward = np.sum(d4rl_data["rewards"], axis=1)

    num_trajs = traj_reward.shape[0]
    sorted_traj_idx = np.argsort(traj_reward)

    num_pos_traj = config["num_positive_trajectories"]
    num_neg_traj = config["num_negative_trajectories"]
    num_union_traj = config["num_union_trajectories"]

    # preferred dataset
    reward_fraction = 0.9
    min_idx = int(reward_fraction * num_trajs)
    pos_idx = sorted_traj_idx[min_idx:]
    num_pos_traj = min(len(pos_idx), num_pos_traj)
    pos_shuffled_idx = np.random.choice(pos_idx, size=num_pos_traj, replace=False)

    # non-preferred dataset
    true_percentage = 1.0 - config["non_pref_noise"]

    num_true_neg_traj = int(num_neg_traj * true_percentage)
    num_false_neg_traj = num_neg_traj - num_true_neg_traj

    reward_fraction = 0.1
    max_idx = int(reward_fraction * num_trajs)
    low_reward_neg_idx = sorted_traj_idx[:max_idx]
    num_true_neg_traj = min(len(low_reward_neg_idx), num_true_neg_traj)

    true_neg_low_reward_idx = np.random.choice(
        low_reward_neg_idx, size=num_true_neg_traj, replace=False
    )
    del_idx = np.concatenate([pos_shuffled_idx, true_neg_low_reward_idx])
    union_idx = np.delete(np.arange(num_trajs), del_idx)
    false_neg_low_reward_idx = np.random.choice(
        union_idx, size=num_false_neg_traj, replace=False
    )
    neg_shuffled_idx = np.concatenate(
        [true_neg_low_reward_idx, false_neg_low_reward_idx]
    )
    np.random.shuffle(neg_shuffled_idx)

    # union dataset
    del_idx = np.concatenate([pos_shuffled_idx, neg_shuffled_idx])
    union_idx = np.delete(np.arange(num_trajs), del_idx)
    num_union_traj = len(union_idx) if num_union_traj < 0 else num_union_traj
    union_shuffled_idx = np.random.choice(union_idx, size=num_union_traj, replace=False)

    print(f"Number of true negative trajectory dataset: {num_true_neg_traj}")
    print(f"Number of false negative trajectory dataset: {num_false_neg_traj}")

    return pos_shuffled_idx, neg_shuffled_idx, union_shuffled_idx


def get_cost_pos_neg_and_union_data(d4rl_data, config):
    traj_cost = np.sum(d4rl_data["costs"], axis=1)

    num_trajs = traj_cost.shape[0]
    sorted_traj_idx = np.argsort(traj_cost)

    num_pos_traj = config["num_positive_trajectories"]
    num_neg_traj = config["num_negative_trajectories"]
    num_union_traj = config["num_union_trajectories"]

    # preferred dataset
    cost_fraction = 0.1
    max_idx = int(cost_fraction * num_trajs)
    pos_idx = sorted_traj_idx[:max_idx]
    num_pos_traj = min(len(pos_idx), num_pos_traj)
    pos_shuffled_idx = np.random.choice(pos_idx, size=num_pos_traj, replace=False)

    # non-preferred dataset
    true_percentage = 1.0 - config["non_pref_noise"]

    num_true_neg_traj = int(num_neg_traj * true_percentage)
    num_false_neg_traj = num_neg_traj - num_true_neg_traj

    cost_fraction = 0.9
    min_idx = int(cost_fraction * num_trajs)
    high_cost_neg_idx = sorted_traj_idx[min_idx:]
    num_true_neg_traj = min(len(high_cost_neg_idx), num_true_neg_traj)

    true_neg_high_cost_idx = np.random.choice(
        high_cost_neg_idx, size=num_true_neg_traj, replace=False
    )
    del_idx = np.concatenate([pos_shuffled_idx, true_neg_high_cost_idx])
    union_idx = np.delete(np.arange(num_trajs), del_idx)
    false_neg_high_cost_idx = np.random.choice(
        union_idx, size=num_false_neg_traj, replace=False
    )
    neg_shuffled_idx = np.concatenate([true_neg_high_cost_idx, false_neg_high_cost_idx])
    np.random.shuffle(neg_shuffled_idx)

    # union dataset
    del_idx = np.concatenate([pos_shuffled_idx, neg_shuffled_idx])
    union_idx = np.delete(np.arange(num_trajs), del_idx)
    num_union_traj = len(union_idx) if num_union_traj < 0 else num_union_traj
    union_shuffled_idx = np.random.choice(union_idx, size=num_union_traj, replace=False)

    print(f"Number of true negative trajectory dataset: {num_true_neg_traj}")
    print(f"Number of false negative trajectory dataset: {num_false_neg_traj}")

    return pos_shuffled_idx, neg_shuffled_idx, union_shuffled_idx


def get_full_pos_neg_and_union_data(d4rl_data, config):
    traj_reward = np.sum(d4rl_data["rewards"], axis=1)
    traj_cost = np.sum(d4rl_data["costs"], axis=1)

    num_trajs = traj_reward.shape[0]
    sorted_reward_idx = np.argsort(traj_reward)
    sorted_cost_idx = np.argsort(traj_cost)

    num_pos_traj = config["num_positive_trajectories"]
    num_neg_traj = config["num_negative_trajectories"]
    num_union_traj = config["num_union_trajectories"]

    # preferred dataset
    reward_fraction = 0.5
    min_idx = int(reward_fraction * sorted_reward_idx.shape[0])
    top_reward_idx = sorted_reward_idx[min_idx:]

    ### sorted(top_reward_idx, key=lambda x: traj_cost[x])
    top_reward_sorted_cost_idx = top_reward_idx[np.argsort(traj_cost[top_reward_idx])]
    cost_fraction = 0.1
    max_idx = int(cost_fraction * top_reward_sorted_cost_idx.shape[0])
    pos_idx = top_reward_sorted_cost_idx[:max_idx]
    num_pos_traj = min(len(pos_idx), num_pos_traj)
    pos_shuffled_idx = np.random.choice(pos_idx, size=num_pos_traj, replace=False)

    # non-preferred dataset
    true_percentage = 1.0 - config["non_pref_noise"]
    num_true_neg_traj = int(num_neg_traj * true_percentage)
    num_false_neg_traj = num_neg_traj - num_true_neg_traj

    ### high cost non-preferred dataset
    cost_fraction = 0.9
    min_idx = int(cost_fraction * num_trajs)
    high_cost_neg_idx = sorted_cost_idx[min_idx:]

    ### low reward non-preferred dataset
    reward_fraction = 0.1
    max_idx = int(reward_fraction * num_trajs)
    low_reward_neg_idx = sorted_reward_idx[:max_idx]

    neg_idx = np.union1d(high_cost_neg_idx, low_reward_neg_idx)
    np.random.shuffle(neg_idx)
    num_true_neg_traj = min(len(neg_idx), num_true_neg_traj)
    true_neg_shuffled_idx = np.random.choice(
        neg_idx, size=num_true_neg_traj, replace=False
    )

    del_idx = np.concatenate([pos_shuffled_idx, true_neg_shuffled_idx])
    union_idx = np.delete(np.arange(num_trajs), del_idx)
    false_neg_shuffled_idx = np.random.choice(
        union_idx, size=num_false_neg_traj, replace=False
    )
    neg_shuffled_idx = np.concatenate([true_neg_shuffled_idx, false_neg_shuffled_idx])
    np.random.shuffle(neg_shuffled_idx)

    # union dataset
    del_idx = np.concatenate([pos_shuffled_idx, neg_shuffled_idx])
    union_idx = np.delete(np.arange(num_trajs), del_idx)
    num_union_traj = len(union_idx) if num_union_traj < 0 else num_union_traj
    union_shuffled_idx = np.random.choice(union_idx, size=num_union_traj, replace=False)

    print(f"Number of true negative trajectory dataset: {num_true_neg_traj}")
    print(f"Number of false negative trajectory dataset: {num_false_neg_traj}")

    return pos_shuffled_idx, neg_shuffled_idx, union_shuffled_idx


def get_pos_neg_and_union_data(d4rl_data, config):
    if config["data_inpaint"] == "full":
        pos_idxs, neg_idxs, union_idxs = get_full_pos_neg_and_union_data(
            d4rl_data, config
        )
    elif config["data_inpaint"] == "reward_only":
        pos_idxs, neg_idxs, union_idxs = get_reward_pos_neg_and_union_data(
            d4rl_data, config
        )
    elif config["data_inpaint"] == "cost_only":
        pos_idxs, neg_idxs, union_idxs = get_cost_pos_neg_and_union_data(
            d4rl_data, config
        )
    else:
        raise ValueError(
            "Please set data-inpaint as one of these: full, reward_only, cost_only"
        )

    has_positive = len(pos_idxs) > 0
    has_negative = len(neg_idxs) > 0

    fig, ax = plt.subplots(figsize=(6, 4))

    traj_rewards = d4rl_data["rewards"].sum(1)
    traj_costs = d4rl_data["costs"].sum(1)

    keys = ["observations", "actions", "rewards", "costs", "terminals", "timeouts"]

    union_data = {k: d4rl_data[k][union_idxs] for k in keys}
    ax.plot(traj_costs[union_idxs], traj_rewards[union_idxs], "o", color="lightblue")
    print(f"Number of union trajectory dataset: {union_data['observations'].shape[0]}")
    union_cost, union_reward = (
        union_data["costs"].sum(1).mean(),
        union_data["rewards"].sum(1).mean(),
    )
    print(f"Avg union trajectory cost/reward: {union_cost:.3f}/{union_reward:.3f}")

    pos_data = None
    if has_positive:
        pos_data = {k: d4rl_data[k][pos_idxs] for k in keys}
        ax.plot(traj_costs[pos_idxs], traj_rewards[pos_idxs], "o", color="darkgreen")
        print(
            f"Number of positive trajectory dataset: {pos_data['observations'].shape[0]}"
        )
        pos_cost, pos_reward = (
            pos_data["costs"].sum(1).mean(),
            pos_data["rewards"].sum(1).mean(),
        )
        print(f"Avg positive trajectory cost/reward: {pos_cost:.3f}/{pos_reward:.3f}")

    neg_data = None
    if has_negative:
        neg_data = {k: d4rl_data[k][neg_idxs] for k in keys}
        ax.plot(traj_costs[neg_idxs], traj_rewards[neg_idxs], "o", color="darkred")
        print(
            f"Number of negative trajectory dataset: {neg_data['observations'].shape[0]}"
        )
        neg_cost, neg_reward = (
            neg_data["costs"].sum(1).mean(),
            neg_data["rewards"].sum(1).mean(),
        )
        print(f"Avg negative trajectory cost/reward: {neg_cost:.3f}/{neg_reward:.3f}")

    ax.set_title("Sampled Dataset for Training")
    ax.set_xlabel("Traj Cost")
    ax.set_ylabel("Traj Reward")
    fig.savefig("./sampled_dataset.png", bbox_inches="tight")

    return pos_data, neg_data, union_data


def get_normalized_data(pos_d4rl_data, neg_d4rl_data, union_d4rl_data):
    has_positive = pos_d4rl_data is not None
    has_negative = neg_d4rl_data is not None

    if has_positive:
        pos_obs = pos_d4rl_data["observations"]

    if has_negative:
        neg_obs = neg_d4rl_data["observations"]

    union_obs = union_d4rl_data["observations"]
    mu_obs = union_obs.mean(axis=(0, 1))
    std_obs = union_obs.std(axis=(0, 1))

    if has_positive:
        pos_d4rl_data["observations"] = np.array((pos_obs - mu_obs) / (std_obs + EP))

    if has_negative:
        neg_d4rl_data["observations"] = np.array((neg_obs - mu_obs) / (std_obs + EP))

    union_d4rl_data["observations"] = np.array((union_obs - mu_obs) / (std_obs + EP))

    return pos_d4rl_data, neg_d4rl_data, union_d4rl_data, mu_obs, std_obs
