import jax.numpy as jnp
from flax import linen as nn
from vae_helpers import parse_layer_string, pad_channels, get_width_settings, Conv1x1, Conv3x3, EncBlock, recon_loss, sample
from quantizer import VectorQuantizerEMA
import hps

class BasicUnit(nn.Module):
    H: hps.Hyperparams
    module_type: str 

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
        assert x.shape[1] == blocks[0][0]
        x = Conv3x3(widths[blocks[0][0]])(x)
        for res, spatial_scale in blocks:  # res is the current resolution, spatial_scale is to generate next resolution
            assert x.shape[1] == res
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            block = EncBlock(H, res, use_3x3, spatial_scale or 1, up=up)
            x = block(x, train=train)
            new_res = x.shape[1]
            new_width = widths[new_res]
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
        self.encoder = BasicUnit(H, 'encoder')
        self.quantizer = VectorQuantizerEMA(
                          embedding_dim=H.vq_dim,
                          num_embeddings=H.codebook_size,
                          commitment_cost=0.25,
                          decay=0.99,
                          cross_replica_axis='batch')
        self.decoder = BasicUnit(H, 'decoder')

    def __call__(self, x, is_training=False):
        x_target = jnp.array(x)
        x = self.encoder(x, train=is_training)
        quant_dict = self.quantizer(x, is_training)
        kl = quant_dict['loss']
        px_z = self.decoder(quant_dict['quantize'], train=is_training)
        loss = recon_loss(px_z, x_target)
        return dict(loss=loss + kl, recon_loss=loss, kl=kl, entropy=jnp.log(quant_dict['perplexity']),
                    entropy_ub=jnp.log(self.H.codebook_size))

    def forward_get_latents(self, x):
        x = self.encoder(x)
        return self.quantizer(x, is_training=False)['encoding_indices'].astype(jnp.int32)

    def forward_samples_set_latents(self, latents):
        latents = self.quantizer(None, is_training=False, encoding_indices=latents)
        px_z = self.decoder(latents)
        return sample(px_z)
