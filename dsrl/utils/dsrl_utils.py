import numpy as np
from numba import njit
from numba.typed import List


@njit
def compute_cost_reward_return(
    rew: np.ndarray,
    cost: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    returns,
    costs,
    starts,
    ends,
) -> np.ndarray:
    # Code from DSRL library
    data_num = rew.shape[0]
    rew_ret, cost_ret = 0, 0
    is_start = True
    for i in range(data_num):
        if is_start:
            starts.append(i)
            is_start = False
        rew_ret += rew[i]
        cost_ret += cost[i]
        if terminals[i] or timeouts[i]:
            returns.append(rew_ret)
            costs.append(cost_ret)
            ends.append(i)
            is_start = True
            rew_ret, cost_ret = 0, 0


def get_trajectory_info(dataset: dict):
    # Code from DSRL library
    # we need to initialize the numba List such that it knows the item type
    returns, costs = List([0.0]), List([0.0])
    # store the start and end indexes of the trajectory in the original data
    starts, ends = List([0]), List([0])
    data_num = dataset["rewards"].shape[0]
    print(f"Total number of data points: {data_num}")
    compute_cost_reward_return(
        dataset["rewards"],
        dataset["costs"],
        dataset["terminals"],
        dataset["timeouts"],
        returns,
        costs,
        starts,
        ends,
    )
    return returns[1:], costs[1:], starts[1:], ends[1:]
