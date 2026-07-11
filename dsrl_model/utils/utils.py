import argparse
from distutils.util import strtobool

import gymnasium


class ActionRepeater(gymnasium.Wrapper, gymnasium.utils.RecordConstructorArgs):
    def __init__(self, env: gymnasium.Env, num_repeats: int, discount: int = 1.0):
        gymnasium.utils.RecordConstructorArgs.__init__(self)
        gymnasium.Wrapper.__init__(self, env)
        self._num_repeats = num_repeats
        self._discount = discount

    def step(self, action):
        # Check the equivalent code from TD-MPC
        reward, cost = 0.0, 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            obs, r, term, trun, infos = self.env.step(action)
            reward += discount * (r or 0.0)
            cost += discount * (infos["cost"] or 0.0)
            discount *= self._discount
            if term or trun:
                break
        infos["cost"] = cost
        return obs, reward, term, trun, infos

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def get_params_norm(params, grads=False):
    total_norm = 0.0
    for p in params:
        if grads:
            total_norm += p.grad.detach().data.norm(2).item()
        else:
            total_norm += p.data.norm(2).item()
    return total_norm


def make_static_config_from_dict(name: str, d: dict):
    from flax import struct

    annotations = {}
    defaults = {}

    for k, v in d.items():
        annotations[k] = type(v)
        defaults[k] = struct.field(
            default=v,
            pytree_node=False,  # made it fixed / immutable ?
        )

    cls = type(
        name,
        (),
        {
            "__annotations__": annotations,
            **defaults,
        },
    )

    return struct.dataclass(cls)


def single_agent_args():
    custom_parameters = [
        {"name": "--seed", "type": int, "default": 0, "help": "Random seed"},
        {
            "name": "--use-eval",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "Use evaluation environment for testing",
        },
        {
            "name": "--save-video",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "Use evaluation environment for testing",
        },
        {
            "name": "--task",
            "type": str,
            "default": "OfflinePointGoal1Gymnasium-v0",
            "help": "The task to run",
        },
        {
            "name": "--experiment",
            "type": str,
            "default": "equal",
            "help": "Experiment name",
        },
        {
            "name": "--log-dir",
            "type": str,
            "default": "dsrl_model/runs",
            "help": "directory to save agent logs",
        },
        {
            "name": "--device",
            "type": str,
            "default": "cpu",
            "help": "The device to run the model on",
        },
        {
            "name": "--device-id",
            "type": int,
            "default": 0,
            "help": "The device id to run the model on",
        },
        {
            "name": "--write-terminal",
            "type": lambda x: bool(strtobool(x)),
            "default": True,
            "help": "Toggles terminal logging",
        },
        {
            "name": "--num-epochs",
            "type": int,
            "default": 1000,
            "help": "Total timesteps of the experiments",
        },
        {
            "name": "--batch-size",
            "type": int,
            "default": 128,
            "help": "The number of steps to run in each environment per policy rollout",
        },
        {
            "name": "--bag-size",
            "type": int,
            "default": 512,
            "help": "The number of elements per bag",
        },
        {
            "name": "--warmup-bc",
            "type": int,
            "default": None,
            "help": "Number of epochs that we want to skip before we start training bc model",
        },
        {
            "name": "--train-horizon",
            "type": int,
            "default": None,
            "help": "The horizon length used for training the models",
        },
        {
            "name": "--lr",
            "type": float,
            "default": 1e-5,  # 1e-3 performs better
            "help": "Default common learning rate for the models",
        },
        {
            "name": "--lmbda",
            "type": float,
            "default": None,  # 1e-3 performs better
            "help": "hyperparameter lambda value",
        },
        {
            "name": "--cost-weight-temp",
            "type": float,
            "default": None,
            "help": "Use default value for cost_weight_temp",
        },
        {
            "name": "--value-weight-temp",
            "type": float,
            "default": None,
            "help": "Use default value for value_weight_temp",
        },
        {
            "name": "--value-weight-limit",
            "type": float,
            "default": None,
            "help": "Use default value for value_weight_limit",
        },
        {
            "name": "--update-priority-buffer",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "To update the priority of the buffer used during sampling",
        },
        {
            "name": "--normalize-observation",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "To normalize the state observation.",
        },
        {
            "name": "--cost-model-path",
            "type": str,
            "default": None,
            "help": "set the path of cost model if it already exists.",
        },
        {
            "name": "--bc-weight-binary",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "To use binary weight for learning bc policy",
        },
        {
            "name": "--use-validation",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "To use validation set to select best cost model for bc training",
        },
        {
            "name": "--act-train-use-logprob",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "To use log_prob from training weighted BC algorithm",
        },
        {
            "name": "--policy-type",
            "type": str,
            "default": "gaussian_mixture",  # vae
            "help": "Type of policy to use for training safe BC algorithm",
        },
        {
            "name": "--dwbc-nu",
            "type": float,
            "default": 0.5,
            "help": "nu value used for dwbc algorithm",
        },
        {
            "name": "--cost-loss-type",
            "type": str,
            "default": "loss_1_bce",  # loss_1, loss_2
            "help": "safemil loss function type",
        },
        {
            "name": "--use-cost-attention",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "use attention based cost model and pred trajectory cost",
        },
        {
            "name": "--use-cost-contrastive",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "use contrastive based cost model",
        },
        {
            "name": "--use-contrastive-loss",
            "type": lambda x: bool(strtobool(x)),
            "default": True,
            "help": "use contrastive based cost learning same as SupContrast paper",
        },
        {
            "name": "--pretrain-cost-contrastive",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "pretrain the cost contrative first then train the linear cost model",
        },
        {
            "name": "--use-td3-style-bc",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "use td3 style policy learning, i.e. add bc plus value term",
        },
        {
            "name": "--num-neg-extra-traj",
            "type": int,
            "default": 10,
            "help": "number of extra negative trajectories used in cost contrastive learning",
        },
        {
            "name": "--use-expected-cost-pref",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "use td3 style policy learning, i.e. add bc plus value term",
        },
        {
            "name": "--bootstrap-lambda",
            "type": float,
            "default": 0.3,
            "help": "bootstrap lambda value for cost preference learning",
        },
        {
            "name": "--bc-weight-temp",
            "type": float,
            "default": 0.5,
            "help": "weighted temperature hyper-parameter for BC",
        },
        {
            "name": "--use-osil-weight",
            "type": lambda x: bool(strtobool(x)),
            "default": True,
            "help": "use osil-style value weight in policy learning",
        },
        {
            "name": "--num-preferred",
            "type": int,
            "default": 0,
            "help": "number of preferred trajectories D_P",
        },
        {
            "name": "--num-non-preferred",
            "type": int,
            "default": 50,
            "help": "number of non-preferred trajectories D_N",
        },
        {
            "name": "--num-union",
            "type": int,
            "default": -1,
            "help": "number of union trajectories D_U",
        },
        {
            "name": "--non-pref-noise",
            "type": float,
            "default": 0.0,
            "help": "fraction of noisy label in the non-preferred dataset",
        },
        {
            "name": "--use-bc-trajectory",
            "type": lambda x: bool(strtobool(x)),
            "default": True,
            "help": "use trajectory weighted bc policy",
        },
        {
            "name": "--use-vonmisesfisher-mode",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "use von Mises Fisher hypershere samples for multimode representation",
        },
        {
            "name": "--data-inpaint",
            "type": str,
            "default": None,
            "help": "set data inpaint tuple from full, reward_only, cost_only",
        },
        {
            "name": "--preferred-label",
            "type": float,
            "default": 1.0,
            "help": "preferred label score for training safecl.",
        },
        {
            "name": "--non-preferred-label",
            "type": float,
            "default": 0.0,
            "help": "non-preferred label score for training safecl.",
        },
        {
            "name": "--embd-freq",
            "type": int,
            "default": None,
            "help": "update the target embedding network frequency.",
        },
        {
            "name": "--embd-size",
            "type": int,
            "default": None,
            "help": "embedding size",
        },
        {
            "name": "--policy-baseline-type",
            "type": str,
            "default": "constant",
            "help": "policy baseline type can be 'constant', 'softer_max', 'mean_std'",
        },
        {
            "name": "--policy-weight-type",
            "type": str,
            "default": "value_based",
            "help": "policy weight type can be 'value_based', 'score_based'",
        },
        {
            "name": "--embedding-model-type",
            "type": str,
            "default": "model_projection",
            "help": "embedding model type can be 'no_projection', 'constant_projection', 'model_projection'",
        },
        {
            "name": "--alpha",
            "type": float,
            "default": 0.9,
            "help": "hyperparam alpha",
        },
        {
            "name": "--use-weight-decay",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "SafeCL whether to use weight decay or percentage based",
        },
    ]
    # Create argument parser
    parser = argparse.ArgumentParser(description="RL Policy")
    for param in custom_parameters:
        param_name = param.pop("name")
        parser.add_argument(param_name, **param)

    # Parse arguments

    args = parser.parse_args()
    cfg_env = {}
    return args, cfg_env
