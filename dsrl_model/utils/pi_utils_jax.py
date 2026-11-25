import optax

from dsrl_model.utils.models_jax import GaussianActor, SafeDiceTanhMixtureActor

PI_ACTOR_TYPES = ("tanh_mixture", "gaussian")
PI_LOSS_TYPES = ("sample_l2", "nll")


def get_actor_cls(pi_actor):
    """Policy class for --pi-actor: 'tanh_mixture' or 'gaussian'."""
    if pi_actor == "gaussian":
        return GaussianActor
    if pi_actor == "tanh_mixture":
        return SafeDiceTanhMixtureActor
    raise ValueError(f"pi_actor must be one of {PI_ACTOR_TYPES}, got {pi_actor}")


def check_policy_config(pi_actor, pi_loss):
    import sys

    assert pi_actor in PI_ACTOR_TYPES, pi_actor
    assert pi_loss in PI_LOSS_TYPES, pi_loss
    if pi_loss == "nll" and pi_actor == "tanh_mixture":
        print(
            "WARNING: --pi-loss nll with the tanh-squashed actor gives very large "
            "losses for dataset actions at +/-1 (atanh blow-up); "
            "consider --pi-actor gaussian.",
            file=sys.stderr,
        )


def policy_bc_loss(policy_model, obs, act, pi_loss):
    """
    Per-sample behaviour-cloning loss of the dataset action `act` at `obs`.
      'sample_l2': L2 between a (reparameterised) sampled policy action and `act`.
      'nll':       negative log-likelihood of `act` (AWR/AWAC/IQL weighted ML).
    `pi_loss` must be a static Python string.
    Returns (loss, pred_act, log_pi), where pred_act/log_pi come from the sampled
    action in both modes, so callers that also need a policy sample can reuse them.
    """
    pred_act, log_pi, *_ = policy_model(obs)
    if pi_loss == "nll":
        loss = -policy_model.get_log_prob(obs, act)
    elif pi_loss == "sample_l2":
        loss = optax.l2_loss(pred_act, act).sum(axis=-1)
    else:
        raise ValueError(f"pi_loss must be one of {PI_LOSS_TYPES}, got {pi_loss}")
    return loss, pred_act, log_pi

