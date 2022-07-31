from functools import partial
import jax 
import jax.numpy as jnp
import flax
from flax import linen as nn
from jax import random
from jax import image
from flax.core import freeze, unfreeze
from einops import repeat
import hps
identity = lambda x: x

def gaussian_kl(mu1, mu2, logsigma1, logsigma2):
    return (-0.5 + logsigma2 - logsigma1
            + 0.5 * (jnp.exp(logsigma1) ** 2 + (mu1 - mu2) ** 2)
            / (jnp.exp(logsigma2) ** 2))

def gaussian_sample(mu, sigma, rng):
    return mu + sigma * random.normal(rng, mu.shape)

Conv1x1 = partial(nn.Conv, kernel_size=(1, 1), strides=(1, 1))
Conv3x3 = partial(nn.Conv, kernel_size=(3, 3), strides=(1, 1), padding='SAME')

def resize(img, shape):
    n, _, _, c = img.shape
    return image.resize(img, (n,) + shape + (c,), 'nearest')

def recon_loss(px_z, x):
    return jnp.abs(px_z - x).mean()

def sample(px_z):
    return jnp.round((jnp.clip(px_z, -1, 1) + 1) * 127.5).astype(jnp.uint8)

class Attention(nn.Module):
    H: hps.Hyperparams

    def setup(self):
        H = self.H
        self.attention = nn.SelfAttention(num_heads=H.num_heads)

    def __call__(self, x):
        res = x
        x = self.attention(normalize(x, self.H)) * np.sqrt(1 / x.shape[-1])
        return x + res

def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, count = ss.split('x')
            layers.extend(int(count) * [(int(res), None)])
        elif 'm' in ss:
            res, mixin = ss.split('m')
            layers.append((int(res), int(mixin)))
        elif 'd' in ss:
            res, down_rate = ss.split('d')
            layers.append((int(res), int(down_rate)))
        else:
            res = int(ss)
            layers.append((res, 1))
    return layers

def pad_channels(t, new_width):
    return jnp.pad(t, (t.ndim - 1) * [(0, 0)] + [(0, new_width - t.shape[-1])])

def get_width_settings(s):
    mapping = {}
    if s:
        for ss in s.split(','):
            k, v = ss.split(':')
            mapping[k] = int(v)
    return mapping

def normalize(x, type=None, train=False):
    if type == 'group':
        return nn.GroupNorm()(x)
    elif type == 'batch':
        return nn.BatchNorm(use_running_average=not train, axis_name='batch')(x)
    else:
        return x

# Want to be able to vary the scale of initialized parameters
def lecun_normal(scale):
    return nn.initializers.variance_scaling(scale, 'fan_in', 'truncated_normal')

class Block(nn.Module):
    block_type: str
    bottleneck_multiple: int
    use_3x3: bool = True
    spatial_scale: int = 1
    up: bool = False

    @nn.compact
    def __call__(self, x, train=True):
        width = x.shape[-1]
        middle_width = width * self.bottleneck_multiple
        assert middle_width.is_integer()
        middle_width = int(middle_width)
        Conv3x3_ = Conv3x3 if self.use_3x3 else Conv1x1
        if self.block_type == 'bottleneck':
            x_ = Conv1x1(middle_width)(nn.gelu(x))
            x_ = Conv3x3_(middle_width)(nn.gelu(x_))
            x_ = Conv3x3_(middle_width)(nn.gelu(x_))
            x_ = Conv1x1(width, kernel_init=lecun_normal(1))(nn.gelu(x_))
        else:
            raise NotImplementedError
        out = x + x_
        if self.spatial_scale > 1:
            if self.up:
                out = repeat(out, 'b h w c -> b (h x) (w y) c', x=self.spatial_scale, y=self.spatial_scale)
            else:
                window_shape = 2 * (self.spatial_scale,)
                out = nn.avg_pool(out, window_shape, window_shape)
        return out

def has_attn(res_, H):
    attn_res = [int(res) for res in H.attn_res.split(',') if len(res) > 0]
    return res_ in attn_res
    
class EncBlock(nn.Module):
    H: hps.Hyperparams
    res: int
    spatial_scale: int = 1
    use_3x3: bool = True
    up: bool = False

    def setup(self):
        H = self.H
        self.pre_layer = Attention(H) if has_attn(self.res, H) else identity
        self.block1 = Block(H.block_type, H.bottleneck_multiple, self.use_3x3, self.spatial_scale, up=self.up)
        
    def __call__(self, x, train=True):
        return self.block1(self.pre_layer(x), train=train)
