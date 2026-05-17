import functools

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tensorflow_probability.substrates import jax as tfp_jax

EPS = 1e-7


def sample_von_mises_fisher_samples(key, mean_direction, concentration, num_samples):
    tfd_jax = tfp_jax.distributions
    dist = tfd_jax.VonMisesFisher(
        mean_direction=mean_direction, concentration=concentration
    )
    samples = dist.sample(seed=key, sample_shape=(num_samples,)).T
    return samples


def gumbel_softmax(key, logits, tau, hard=False):
    g = distrax.Gumbel(loc=0.0, scale=1.0).sample(seed=key, sample_shape=logits.shape)
    ysoft = jax.nn.softmax((g + logits) / tau, axis=-1)

    if not hard:
        return ysoft

    yhard = jax.nn.one_hot(jnp.argmax(ysoft, axis=-1), ysoft.shape[1])
    yhard = yhard + ysoft - jax.lax.stop_gradient(ysoft)
    return yhard


def l2_normalize(x, axis=None, eps=EPS):
    return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


def log1pexp(x, eps=EPS):
    # safe implementation of L = log(1 + exp(x))
    # x > 0: L = x + log(1 + exp(-x))
    # x <=0: L = log(1 + exp(x))
    # combined: L = Relu(x) + log(1 + exp(-|x|)) = jax.nn.softplus(x)
    abs_x = jnp.abs(x)
    pos_x = jax.nn.relu(x)
    return pos_x + jnp.log(1 + jnp.exp(-abs_x))


def get_tree_norm(tree):
    square_tree = jax.tree_util.tree_map(lambda x: jnp.sum(x**2), tree)
    total_square = jax.tree_util.tree_reduce(lambda acc, x: acc + x, square_tree)
    l2_norm = jnp.sqrt(total_square)
    return l2_norm


def bce_loss(logits, labels, weights=1.0):
    """
    Numerically Stable BCE loss
    Doc: https://medium.com/@sahilcarterr/why-nn-bcewithlogitsloss-numerically-stable-6a04f3052967
    """
    tn = jnp.clip(-logits, min=0.0)
    loss = (1 - labels) * logits + tn + jnp.logaddexp(-tn, -logits - tn)
    loss = weights * loss
    return jnp.mean(loss)


class Scalar(nnx.Module):
    def __init__(self, val):
        dtype = jnp.float32
        self.val = nnx.Param(jnp.array(val, dtype=dtype))

    def __call__(self):
        return self.val


class SafeDiceTanhMixtureActor(nnx.Module):
    def __init__(
        self,
        rngs,
        obs_dim,
        act_dim,
        hidden_size=256,
        num_components=2,
        mean_range=(-5.0, 5.0),
        logstd_range=(-5.0, 1.0),
        eps=EPS,
        mdn_temperature=1.0,
    ):
        self.rngs = rngs

        self.act_dim = act_dim
        self.num_components = num_components
        self.mdn_temp = mdn_temperature

        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

        self.pre_encoder = nnx.Sequential(
            nnx.Linear(obs_dim, hidden_size, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.relu,
        )

        self.logits = nnx.Linear(hidden_size, num_components, rngs=rngs)
        self.means = nnx.Linear(hidden_size, num_components * act_dim, rngs=rngs)
        self.logstds = nnx.Linear(hidden_size, num_components * act_dim, rngs=rngs)

    def get_pretanh_action_dist(self, obs):
        x = self.pre_encoder(obs)

        mixture_logits = self.logits(x) / self.mdn_temp
        means = jnp.clip(self.means(x), self.mean_min, self.mean_max)
        means = means.reshape(-1, self.num_components, self.act_dim)
        logstds = jnp.clip(self.logstds(x), self.logstd_min, self.logstd_max)
        logstds = logstds.reshape(-1, self.num_components, self.act_dim)
        stds = jnp.exp(logstds)

        mixture_dist = distrax.Categorical(logits=mixture_logits)
        component_dist = distrax.Normal(loc=means, scale=stds)
        component_dist = distrax.Independent(component_dist, 1)
        pretanh_action_dist = distrax.MixtureSameFamily(mixture_dist, component_dist)

        return pretanh_action_dist

    def __call__(self, obs):
        pretanh_action_dist = self.get_pretanh_action_dist(obs)

        pretanh_actions = pretanh_action_dist.sample(seed=self.rngs())
        actions = jnp.tanh(pretanh_actions)

        # mixture_sample = gumbel_softmax(self.rngs(), mixture_logits, tau=1.0, hard=True)
        # component_sample = component_dist.sample(seed=self.rngs())
        # pretanh_actions = jnp.einsum("ij,ijk->ik", mixture_sample, component_sample)
        # actions = jax.nn.tanh(pretanh_actions)

        pretanh_logp = pretanh_action_dist.log_prob(pretanh_actions)
        # jacobian_det = jnp.sum(jnp.log(1 - actions**2 + self.eps), axis=-1)
        jacobian_det = jnp.sum(
            2.0
            * (jnp.log(2.0) - pretanh_actions - nnx.softplus(-2.0 * pretanh_actions)),
            axis=-1,
        )
        log_prob = pretanh_logp - jacobian_det

        return actions, log_prob, pretanh_actions, pretanh_action_dist

    def get_log_prob(self, obs, act):
        act = jnp.clip(act, -1.0 + self.eps, 1.0 - self.eps)
        pretanh_act = jnp.atanh(act)

        pretanh_dist = self.get_pretanh_action_dist(obs)
        pretanh_logp = pretanh_dist.log_prob(pretanh_act)
        # jacobian_det = jnp.sum(jnp.log(1 - actions**2 + self.eps), axis=-1)
        jacobian_det = jnp.sum(
            2.0 * (jnp.log(2.0) - pretanh_act - nnx.softplus(-2.0 * pretanh_act)),
            axis=-1,
        )
        log_prob = pretanh_logp - jacobian_det

        return log_prob

    @functools.partial(jax.jit, static_argnums=0)
    def action_w_key(self, key, obs, deterministic=False):
        x = self.pre_encoder(obs)

        mixture_logits = self.logits(x) / self.mdn_temp
        means = jnp.clip(self.means(x), self.mean_min, self.mean_max)
        means = means.reshape(-1, self.num_components, self.act_dim)
        logstds = jnp.clip(self.logstds(x), self.logstd_min, self.logstd_max)
        logstds = logstds.reshape(-1, self.num_components, self.act_dim)
        stds = jnp.exp(logstds)

        mixture_dist = distrax.Categorical(logits=mixture_logits)

        def deterministic_fun():
            mixture_id = mixture_dist.sample(seed=key)
            pretanh_action = jax.vmap(lambda x, y: x[y])(means, mixture_id)
            return pretanh_action

        def stochastic_fun():
            component_dist = distrax.Normal(loc=means, scale=stds)
            component_dist = distrax.Independent(component_dist, 1)
            pretanh_action_dist = distrax.MixtureSameFamily(
                mixture_dist, component_dist
            )
            pretanh_action = pretanh_action_dist.sample(seed=key)
            return pretanh_action

        pretanh_action = jnp.where(deterministic, deterministic_fun(), stochastic_fun())

        return jax.nn.tanh(pretanh_action)

    def action(self, obs, deterministic=False):
        return self.action_w_key(self.rngs(), obs, deterministic)


class ExpCostModel(nnx.Module):
    def __init__(self, rngs, x_dims, hidden_size=256, clip_range=(0.0, 1.0)):
        sizes = [x_dims, hidden_size, hidden_size, 1]
        layers = list()
        for j in range(len(sizes) - 1):
            act = nnx.elu if j < len(sizes) - 2 else jax.nn.identity
            affine_layer = nnx.Linear(sizes[j], sizes[j + 1], rngs=rngs)
            layers += [affine_layer, act]
        self.model = nnx.Sequential(*layers)
        self.min, self.max = clip_range

    def __call__(self, x):
        x = jnp.squeeze(self.model(x), axis=-1)
        ret = jax.nn.sigmoid(x)
        ret = jnp.clip(ret, min=self.min, max=self.max)
        return ret


class ContrastiveCostModel(nnx.Module):
    def __init__(self, rngs, x_dims, hidden_size=256):
        sizes = [x_dims, hidden_size, hidden_size, 128]
        layers = list()
        for j in range(len(sizes) - 1):
            act = nnx.elu if j < len(sizes) - 2 else jax.nn.identity
            affine_layer = nnx.Linear(sizes[j], sizes[j + 1], rngs=rngs)
            layers += [affine_layer, act]
        self.encoder = nnx.Sequential(*layers)
        self.projection = nnx.Linear(sizes[-1], 1, rngs=rngs)

    def __call__(self, x):
        z = l2_normalize(self.encoder(x), axis=-1)
        proj_z = jnp.squeeze(self.projection(z), axis=-1)
        cost = jax.nn.sigmoid(proj_z)
        return z, cost


class TdmpcValue(nnx.Module):
    def __init__(self, rngs, x_dim, hidden_size=256):
        zero_init = nnx.initializers.zeros

        self.model = nnx.Sequential(
            nnx.Linear(x_dim, hidden_size, rngs=rngs),
            nnx.LayerNorm(hidden_size, rngs=rngs),
            nnx.elu,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.elu,
            nnx.Linear(
                hidden_size, 1, kernel_init=zero_init, bias_init=zero_init, rngs=rngs
            ),
        )

    def __call__(self, x):
        return jnp.squeeze(self.model(x), axis=-1)


class EnsembleValue(nnx.Module):
    def __init__(self, rngs, x_dim, hidden_size=256):
        self.v1 = TdmpcValue(rngs, x_dim, hidden_size)
        self.v2 = TdmpcValue(rngs, x_dim, hidden_size)

    def __call__(self, x):
        return self.v1(x), self.v2(x)


def positionalencoding1d(d_model, length):
    """
    Code: https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
    """
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dim (got dim={:d})".format(d_model)
        )
    pe = np.zeros((length, d_model), dtype=np.float32)
    position = np.expand_dims(np.arange(0, length), axis=1)
    div_term = np.exp((np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = np.expand_dims(pe, axis=0)
    return jnp.array(pe)


class TransformerBlock(nnx.Module):
    """
    Code: https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html
    """

    def __init__(
        self,
        rngs,
        d_model,
        liner_features,
        num_heads,
        do_layer_norm=True,
        do_residual=True,
        rate=0.3,
    ):
        self.attn = nnx.MultiHeadAttention(num_heads, d_model, decode=False, rngs=rngs)
        self.dp1 = nnx.Dropout(rate=rate, rngs=rngs)
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ff = nnx.Sequential(
            nnx.Linear(d_model, liner_features, rngs=rngs),
            nnx.gelu,
            nnx.Linear(liner_features, d_model, rngs=rngs),
        )
        self.dp2 = nnx.Dropout(rate=rate, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.do_layer_norm = do_layer_norm
        self.do_residual = do_residual

    def maybe_layer_norm(self, ln, x):
        if self.do_layer_norm:
            x = ln(x)
        return x

    def maybe_residual(self, prev_x, x):
        if self.do_residual:
            x = prev_x + x
        return x

    def __call__(self, x, mask=None, training=False):
        # x shape: batch x horizon x d_model

        # layer norm: ln1(x)
        ln_x = self.maybe_layer_norm(self.ln1, x)
        attn_x = self.dp1(self.attn(ln_x, mask=mask), deterministic=not training)
        # residual connection: x + attn_x
        x = self.maybe_residual(x, attn_x)

        # layer norm: ln2(x)
        ln_x = self.maybe_layer_norm(self.ln2, x)
        ff_x = self.dp2(self.ff(ln_x), deterministic=not training)
        # residual connection: x + ff_x
        x = self.maybe_residual(x, ff_x)
        return x


class TransformerEmbedding(nnx.Module):
    def __init__(
        self,
        rngs,
        obs_dim,
        act_dim,
        horizon,
        embd_dim=128,
        num_heads=4,
        num_attentions=1,
        do_layer_norm=True,
        do_residual=True,
    ):
        self.horizon = horizon
        self.encoder = nnx.Linear(obs_dim + act_dim, embd_dim, rngs=rngs)
        # self.pos_encoding = positionalencoding1d(embd_dim, horizon)
        self.pos_embedding = nnx.Embed(
            num_embeddings=horizon, features=embd_dim, rngs=rngs
        )
        self.mask = jnp.tril(jnp.ones((horizon, horizon)))  # causal mask
        linear_features = 4 * embd_dim
        self.transformer_blocks = tuple(
            [
                TransformerBlock(
                    rngs=rngs,
                    d_model=embd_dim,
                    liner_features=linear_features,
                    num_heads=num_heads,
                    do_layer_norm=do_layer_norm,
                    do_residual=do_residual,
                )
                for _ in range(num_attentions)
            ]
        )
        self.attention = nnx.Linear(embd_dim, 1, rngs=rngs)

    def z(self, x, normalize_z=True, training=True):
        # ensure x: batch X horizon X obs_act_dim
        x = self.encoder(x)
        positions = jnp.arange(x.shape[1])[None, :]  # (1, Horizon)
        x += self.pos_embedding(positions)
        for transformer in self.transformer_blocks:
            x = transformer(x, self.mask, training)

        if normalize_z:
            x = l2_normalize(x, axis=-1)

        # batch X horizon X embd_dim
        return x

    def get_score(self, x, ztarget, normalize_z=True, training=True):
        # batch X horizon X embd_dim
        z = self.z(x, normalize_z, training)
        # batch X horizon
        traj_score = jnp.einsum("ijk,ijk->ij", z, ztarget)
        attn_logits = self.attention(z).squeeze(axis=-1)
        # mask first half of the trajectory
        first_half = jnp.arange(self.horizon) <= self.horizon // 2
        attn_logits = attn_logits + first_half * -1e9
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        # batch
        score = jnp.einsum("ij,ij->i", attn_weights, traj_score)
        return score
