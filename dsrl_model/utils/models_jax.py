import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

EPS = 1e-7


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


class SafeDiceTanhMixtureActor(nnx.Module):
    def __init__(
        self,
        rngs,
        obs_dim,
        act_dim,
        hidden_size=256,
        num_components=2,
        mean_range=(-7.0, 7.0),
        logstd_range=(-5.0, 2.0),
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

    def __call__(self, obs):
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

        mixture_sample = gumbel_softmax(self.rngs(), mixture_logits, tau=1.0, hard=True)
        component_sample = component_dist.sample(seed=self.rngs())
        pretanh_actions = jnp.einsum("ij,ijk->ik", mixture_sample, component_sample)
        actions = jax.nn.tanh(pretanh_actions)

        return actions, pretanh_actions, pretanh_action_dist

    def action(self, obs, deterministic=False):
        x = self.pre_encoder(obs)

        mixture_logits = self.logits(x) / self.mdn_temp
        means = jnp.clip(self.means(x), self.mean_min, self.mean_max)
        means = means.reshape(-1, self.num_components, self.act_dim)
        logstds = jnp.clip(self.logstds(x), self.logstd_min, self.logstd_max)
        logstds = logstds.reshape(-1, self.num_components, self.act_dim)
        stds = jnp.exp(logstds)

        mixture_dist = distrax.Categorical(logits=mixture_logits)

        if deterministic:
            mixture_id = mixture_dist.sample(seed=self.rngs())
            pretanh_action = jax.vmap(lambda x, y: x[y])(means, mixture_id)
        else:
            component_dist = distrax.Normal(loc=means, scale=stds)
            component_dist = distrax.Independent(component_dist, 1)
            pretanh_action_dist = distrax.MixtureSameFamily(
                mixture_dist, component_dist
            )
            pretanh_action = pretanh_action_dist.sample(seed=self.rngs())

        return jax.nn.tanh(pretanh_action)


class ExpCostModel(nnx.Module):
    def __init__(self, rngs, x_dims, hidden_size=256):
        sizes = [x_dims, hidden_size, hidden_size, 1]
        layers = list()
        for j in range(len(sizes) - 1):
            act = nnx.elu if j < len(sizes) - 2 else jax.nn.identity
            affine_layer = nnx.Linear(sizes[j], sizes[j + 1], rngs=rngs)
            layers += [affine_layer, act]
        self.model = nnx.Sequential(*layers)

    def __call__(self, x):
        x = jnp.squeeze(self.model(x), axis=-1)
        return jax.nn.sigmoid(x)


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

    def __init__(self, rngs, d_model, liner_features, num_heads, rate=0.1):
        self.attn = nnx.MultiHeadAttention(num_heads, d_model, decode=False, rngs=rngs)
        self.dp1 = nnx.Dropout(rate=rate, rngs=rngs)
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ff = nnx.Sequential(
            nnx.Linear(d_model, liner_features, rngs=rngs),
            nnx.relu,
            nnx.Linear(liner_features, d_model, rngs=rngs),
        )
        self.dp2 = nnx.Dropout(rate=rate, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(self, x, mask=None, training=False):
        # x shape: batch x horizon x d_model
        attn_x = self.dp1(self.attn(x, mask=mask), deterministic=not training)
        # residual connection
        x = x + attn_x
        x = self.ln1(x)

        ff_x = self.dp2(self.ff(x), deterministic=not training)
        # residual connection
        x = x + ff_x
        x = self.ln2(x)
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
    ):
        self.encoder = nnx.Linear(obs_dim + act_dim, embd_dim, rngs=rngs)
        self.pos_encoding = positionalencoding1d(embd_dim, horizon)
        self.mask = jnp.tril(jnp.ones((horizon, horizon)))  # causal mask
        linear_features = 4 * embd_dim
        self.transformer_blocks = [
            TransformerBlock(
                rngs=rngs,
                d_model=embd_dim,
                liner_features=linear_features,
                num_heads=num_heads,
            )
            for _ in range(num_attentions)
        ]

    def __call__(self, x, normalize_z=True, training=True):
        # ensure x: batch X horizon X obs_act_dim
        x = self.encoder(x)
        x += self.pos_encoding
        for transformer in self.transformer_blocks:
            x = transformer(x, self.mask, training)

        x = x[:, -1, :]
        if normalize_z:
            x = l2_normalize(x, axis=-1)

        return x
