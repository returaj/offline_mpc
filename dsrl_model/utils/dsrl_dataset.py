import numpy as np
import dsrl.infos as dsrl_infos


def get_post_processed_dataset(env, data, config, task):
    density = config["density"]
    cbins, rbins = 10, 50
    max_npb, min_npb = 30, 2
    if density < 1.0:
        density_cfg = dsrl_infos.DENSITY_CFG[task + "_density" + str(density)]
        cbins, rbins = density_cfg["cbins"], density_cfg["rbins"]
        max_npb, min_npb = density_cfg["max_npb"], density_cfg["min_npb"]

    for inpaint_range in config["inpaint_ranges"]:
        data = env.pre_process_data(
            data_dict=data,
            inpaint_ranges=(inpaint_range,),
            density=density,
            cbins=cbins,
            rbins=rbins,
            max_npb=max_npb,
            min_npb=min_npb,
        )
    return data


def to_d4rl_format(data):
    dones_idx = np.where((data["terminals"] == 1) | (data["timeouts"] == 1))[0]
    d4rl_data = {k: [] for k in data.keys()}
    for i in range(dones_idx.shape[0]):
        start = 0 if i == 0 else dones_idx[i - 1] + 1
        end = dones_idx[i] + 1
        for k, v in data.items():
            d4rl_data[k].append(v[start:end])
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


def get_dataset_in_d4rl_format(env, config, task, num_folds=1):
    data = env.get_dataset()
    data = get_post_processed_dataset(env, data, config, task)

    d4rl_data = to_d4rl_format(data)
    keys = ["observations", "actions", "rewards", "costs"]
    return {k: fold_sa_pair(d4rl_data[k], num_folds) for k in keys}
