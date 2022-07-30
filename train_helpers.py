# import torch
import numpy as np
import dataclasses
import argparse
import os
import subprocess
import time

from jax.interpreters.xla import DeviceArray
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import logger, mkdir_p
from vqvae import VQVAE

import jax
from jax import random
import jax.numpy as jnp
from jax.util import safe_map
from flax import jax_utils
from flax.training import checkpoints
from flax.optim import Adam
from functools import partial
from PIL import Image
from jax import lax, pmap
from vae_helpers import sample
import input_pipeline
from einops import rearrange
map = safe_map


def save_model(path, optimizer, ema, state, H):
    optimizer = jax_utils.unreplicate(optimizer)
    step = optimizer.state.step if not H.gan else optimizer['G'].state.step  
    checkpoints.save_checkpoint(path, optimizer, step)
    if ema:
        ema = jax_utils.unreplicate(ema)
        checkpoints.save_checkpoint(path + '_ema', ema, step)
    if state:
        state = jax_utils.unreplicate(state)
        checkpoints.save_checkpoint(path + '_state', state, step)
    from_log = os.path.join(H.save_dir, 'log.jsonl')
    to_log = f'{os.path.dirname(path)}/{os.path.basename(path)}-log.jsonl'
    subprocess.check_output(['cp', from_log, to_log])


def load_vaes(H, logprint):
    rng = random.PRNGKey(H.seed_init)
    init_rng, init_eval_rng = random.split(rng)
    init_eval_rng, init_emb_rng = random.split(init_eval_rng)
    init_batch = jnp.zeros((1, H.image_size, H.image_size, H.n_channels))
    variables = VQVAE(H).init({'params': init_rng}, init_batch, rng=init_eval_rng)
    state, params = variables.pop('params')
    #print(jax.tree_map(jnp.shape, state))
    del variables
    ema = params if H.ema_rate != 0 else {}
    optimizer = Adam(weight_decay=H.wd, beta1=H.adam_beta1,
                     beta2=H.adam_beta2).create(params)
    
    if H.restore_path and H.restore_iter > 0:
        logprint(f'Restoring vae from {H.restore_path}')
        optimizer = checkpoints.restore_checkpoint(H.restore_path, optimizer, step=H.restore_iter)
        if ema:
            ema = checkpoints.restore_checkpoint(H.restore_path + '_ema', ema, step=H.restore_iter)
        if state:
            state = checkpoints.restore_checkpoint(H.restore_path + '_state', state, step=H.restore_iter)

    total_params = 0
    for p in jax.tree_flatten(optimizer.target)[0]:
        total_params += np.prod(p.shape)
    logprint(total_params=total_params, readable=f'{total_params:,}')
    optimizer = jax_utils.replicate(optimizer)
    if ema:
        ema = jax_utils.replicate(ema)        
    if state:
        state = jax_utils.replicate(state)
    return optimizer, ema, state

def accumulate_stats(stats, frequency):
    z = {}
    for k in stats[-1]:
        if k in ['loss_nans', 'kl_nans', 'skipped_updates']:
            z[k] = np.sum([a[k] for a in stats[-frequency:]])
        elif k == 'grad_norm':
            vals = [a[k] for a in stats[-frequency:]]
            finites = np.array(vals)[np.isfinite(vals)]
            if len(finites) == 0:
                z[k] = 0.0
            else:
                z[k] = np.max(finites)
        elif k == 'loss':
            vals = [a[k] for a in stats[-frequency:]]
            finites = np.array(vals)[np.isfinite(vals)]
            z['loss'] = np.mean(vals)
            z['loss_filtered'] = np.mean(finites)
        elif k == 'iter_time':
            z[k] = (stats[-1][k] if len(stats) < frequency
                    else np.mean([a[k] for a in stats[-frequency:]]))
        else:
            z[k] = np.mean([a[k] for a in stats[-frequency:]])
    return z

def linear_warmup(warmup_iters):
    return lambda i: lax.min(1., i / warmup_iters)

def setup_save_dirs(H):
    save_dir = os.path.join(H.save_dir, H.desc)
    mkdir_p(save_dir)
    logdir = os.path.join(save_dir, 'log')
    return dataclasses.replace(
        H,
        save_dir=save_dir,
        logdir=logdir,
    )

def set_up_hyperparams(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()
    parser = add_vae_arguments(parser)
    H = parse_args_and_update_hparams(H, parser, s=s)
    H = setup_save_dirs(H)
    log = logger(H.logdir)
    if H.log_wandb:
        import wandb
        def logprint(*args, pprint=False, **kwargs):
            if len(args) > 0: log(*args)
            wandb.log({k: np.array(x) if type(x) is DeviceArray else x for k, x in kwargs.items()})
        wandb.init(project='vae', entity=H.entity, name=H.name, config=dataclasses.asdict(H))
    else:
        logprint = log
        for i, k in enumerate(sorted(dataclasses.asdict(H))):
            logprint(type='hparam', key=k, value=getattr(H, k))
    np.random.seed(H.seed)
    logprint('training model', H.desc, 'on', H.dataset)
    H = dataclasses.replace(
        H,
        seed_init  =H.seed,
        seed_sample=H.seed + 1,
        seed_train =H.seed + 2 + H.host_id,
        seed_eval  =H.seed + 2 + H.host_count + H.host_id,
    )
    return H, logprint

def clip_grad_norm(g, max_norm):
    # Simulates torch.nn.utils.clip_grad_norm_
    g, treedef = jax.tree_flatten(g)
    total_norm = jnp.linalg.norm(jnp.array(map(jnp.linalg.norm, g)))
    clip_coeff = jnp.minimum(max_norm / (total_norm + 1e-6), 1)
    g = [clip_coeff * g_ for g_ in g]
    return treedef.unflatten(g), total_norm

def write_images(H, optimizer, ema, state, viz_batch):
    params = ema or optimizer.target
    ema_apply = partial(VQVAE(H).apply,
                        {'params': params, **state}) 
    forward_get_latents = partial(ema_apply, method=VQVAE(H).forward_get_latents)
    forward_samples_set_latents = partial(
        ema_apply, method=VQVAE(H).forward_samples_set_latents)

    batches = [sample(viz_batch)]
    zs = forward_get_latents(viz_batch)
    batches.append(forward_samples_set_latents(zs))
    im = jnp.stack(batches)
    return im

def p_write_images(H, optimizer, ema, state, ds, fname, logprint):
    for x in input_pipeline.prefetch(ds, n_prefetch=2):
        viz_batch = x['image']
        fun = pmap(write_images, 'batch', static_broadcasted_argnums=0)
        im = np.array(fun(H, optimizer, ema, state, viz_batch))
        im = rearrange(im, 'device height batch ... -> (device batch) height ...')[:H.num_images_visualize]
        im = rearrange(im, 'batch height h w c -> (height h) (batch w) c')
        logprint(f'printing samples to {fname}')
        Image.fromarray(im).save(fname)
        break
