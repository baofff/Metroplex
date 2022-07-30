from functools import partial
import jax.numpy as jnp
from flax import linen as nn
from vae_helpers import parse_layer_string, pad_channels, get_width_settings, Conv1x1, Conv3x3, EncBlock, checkpoint, recon_loss, sample
from quantizer import VectorQuantizerEMA
import hps
import numpy as np

class BasicUnit(nn.Module):
    H: hps.Hyperparams
    module_type: str 
    min_res: int = 1

    @nn.compact
    def __call__(self, x, train=False):
        H = self.H
        module_type = self.module_type
        if module_type == "encoder":
            block_str = H.enc_blocks
            up = False
        elif module_type == "decoder":
            block_str = H.dec_blocks
            up = True
        else:
            raise NotImplementedError
        widths = get_width_settings(H.custom_width_str)
        blocks = parse_layer_string(block_str)
        x = Conv3x3(widths[str(blocks[0][0])])(x)
        for res, down_rate in blocks:
            if res < self.min_res:
                continue
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            width = widths.get(str(res), H.width)
            block = EncBlock(H, res, width, down_rate or 1, use_3x3, last_scale=np.sqrt(1 / len(blocks)), up=up)
            x = checkpoint(partial(block.__call__, train=train), H, (x,)) #TODO: needs to be fixed for batchnorm
            new_res = x.shape[1]
            new_width = widths.get(str(new_res), H.width)
            if x.shape[3] < new_width:
                x = pad_channels(x, new_width)
            elif x.shape[3] > new_width:
                x = x[..., :new_width]
        if module_type == 'decoder':
            x = Conv1x1(H.n_channels)(x)
        return x
        

class VQVAE(nn.Module):
    H: hps.Hyperparams

    def setup(self):
        H = self.H
        widths = get_width_settings(H.custom_width_str)
        vq_dim = widths[str(H.vq_res)]
        self.encoder = BasicUnit(H, 'encoder', min_res=H.vq_res)
        self.quantizer = VectorQuantizerEMA(
                          embedding_dim=vq_dim,
                          num_embeddings=H.codebook_size,
                          commitment_cost=0.25,
                          decay=0.99,
                          cross_replica_axis='batch')
        self.decoder = BasicUnit(H, 'decoder', min_res=H.vq_res)

    def __call__(self, x, is_training=False):
        x_target = jnp.array(x)
        x = self.encoder(x, train=is_training)
        quant_dict = self.quantizer(x, is_training)
        kl = quant_dict['loss']
        px_z = self.decoder(quant_dict['quantize'], train=is_training)
        loss = recon_loss(px_z, x_target)
        return dict(loss=loss + kl, recon_loss=loss, kl=kl)

    def forward_get_latents(self, x):
        x = self.encoder(x)
        return self.quantizer(x, is_training=False)['encoding_indices'].astype(jnp.int32)

    def forward_samples_set_latents(self, latents):
        latents = self.quantizer(None, is_training=False, encoding_indices=latents)
        px_z = self.decoder(latents)
        return sample(px_z)
