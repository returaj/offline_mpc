import numpy as np
import dsrl.infos as dsrl_infos


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
                repeat_val = np.tile(val[-1], (repeat_len, *other_dim))
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


def get_neg_and_union_data(d4rl_data, config):
    traj_cost = np.sum(d4rl_data["costs"], axis=1)

    num_neg_traj = config["num_negative_trajectories"]
    num_uneg_traj = config["num_union_negative_trajectories"]
    num_upos_traj = config["num_union_positive_trajectories"]

    neg_idx = np.where(traj_cost > 75.0)[0]
    pos_idx = np.where(traj_cost < 25.0)[0]

    keys = ["observations", "actions", "rewards", "costs", "terminals", "timeouts"]
    neg_data = {k: d4rl_data[k][neg_idx[:num_neg_traj]] for k in keys}
    union_neg_data = {
        k: d4rl_data[k][neg_idx[num_neg_traj : num_neg_traj + num_uneg_traj]]
        for k in keys
    }
    union_pos_data = {k: d4rl_data[k][pos_idx[:num_upos_traj]] for k in keys}

    print(f"Number of negative trajectory dataset: {neg_data['observations'].shape[0]}")
    neg_cost, neg_reward = (
        neg_data["costs"].sum(1).mean(),
        neg_data["rewards"].sum(1).mean(),
    )
    print(f"Avg negative trajectory cost/reward: {neg_cost:.3f}/{neg_reward:.3f}")
    print(
        f"Number of union negative trajectory dataset: {union_neg_data['observations'].shape[0]}"
    )
    print(
        f"Number of union positive trajectory dataset: {union_pos_data['observations'].shape[0]}"
    )

    union_data = {
        k: np.concatenate([union_neg_data[k], union_pos_data[k]], axis=0) for k in keys
    }
    union_cost, union_reward = (
        union_data["costs"].sum(1).mean(),
        union_data["rewards"].sum(1).mean(),
    )
    print(f"Avg union trajectory cost/reward: {union_cost:.3f}/{union_reward:.3f}")

    return neg_data, union_data


def get_normalized_data(neg_d4rl_data, union_d4rl_data):
    neg_obs, union_obs = neg_d4rl_data["observations"], union_d4rl_data["observations"]
    mu_obs = (
        neg_obs.mean(axis=(0, 1)) * neg_obs.shape[0]
        + union_obs.mean(axis=(0, 1)) * union_obs.shape[0]
    ) / (neg_obs.shape[0] + union_obs.shape[0])
    std_obs = (
        neg_obs.std(axis=(0, 1)) * neg_obs.shape[0]
        + union_obs.std(axis=(0, 1)) * union_obs.shape[0]
    ) / (neg_obs.shape[0] + union_obs.shape[0])
    neg_d4rl_data["observations"] = np.array((neg_obs - mu_obs) / (std_obs + EP))
    union_d4rl_data["observations"] = np.array((union_obs - mu_obs) / (std_obs + EP))
    return neg_d4rl_data, union_d4rl_data, mu_obs, std_obs
