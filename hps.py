import dataclasses
import jax
import json

@dataclasses.dataclass(frozen=True)
class Hyperparams:
    # general
    hps: str = None
    run_opt: str = 'train'
    model: str = 'vqvae'
    desc: str = 'test' # description of run
    device_count: int = jax.local_device_count()
    host_count: int = jax.host_count()
    host_id: int = jax.host_id()
        
    # optimization
    adam_beta1: float = .9
    adam_beta2: float = .9
    lr: float = .0003
    ema_rate: float = 0.
    n_batch: int = 32   
    warmup_iters: float = 100.
    wd: float = 0.
    grad_clip: float = 200.

    # training misc.
    iters_per_ckpt: int = 25000
    iters_per_images: int = 100
    iters_per_print: int = 10
    iters_per_save: int = 10000
        
    # architecture -------------------------
    
    # for all settings
    custom_width_str: str = ''
    codebook_size: int = None
    vq_dim: int = 256

    # for vq/vdvae only
    block_type: str = "bottleneck"
    attn_res: str = ''
    bottleneck_multiple: float = 0.25
    enc_blocks: str = None
    dec_blocks: str = None
    '''
    Example:
    
    VQVAE (d = up (decoder) or down (encoder)) 
    "dec_blocks": "32x1,32d2,64x1,64d2,128x1,128d2,256x2",
    "enc_blocks": "256x1,256d2,128x1,128d2,64x1,64d2,32x2",   

    '''
    
    # -------------------------------------
    
    # visualization
    num_images_visualize: int = 6
    num_variables_visualize: int = 6
    
    # dataset
    n_channels: int = 3
    image_size: int = None
    split_train: str = 'train'
    split_test: str = 'test'
    data_root: str = './'
    dataset: str = None
    shuffle_buffer: int = 50000
    tfds_data_dir: str = None
    #tfds_data_dir: Optional directory where tfds datasets are stored. If not
    #  specified, datasets are downloaded and in the default tfds data_dir on the
    #  local machine.
    tfds_manual_dir: str = None # Path to manually downloaded dataset
    '''
    The only preprocessing implemented rn is center crop and then resizing. 
    But this can be easily modified if necessary.
    '''
       
    # log 
    logdir: str = None
    log_wandb: bool = False
    project: str = 'vae'
    entity: str = None
    name: str = None
    early_evals: int = 0
    
    # save & restore
    save_dir: str = './saved_models'
    restore_path: str = None
    restore_iter: int = 0 # setting this to 0 = new run from scratch
    
    # seed
    seed: int = 0


def parse_args_and_update_hparams(H, parser, s=None):
    parser_dict = vars(parser.parse_args(s))
    json_file = parser_dict['hps']
    with open(json_file) as f:
        json_dict = json.load(f)
    parser_dict.update(json_dict)
    return dataclasses.replace(H, **json_dict)

def add_vae_arguments(parser):
    for f in dataclasses.fields(Hyperparams):
        kwargs = (dict(action='store_true') if f.type is bool and not f.default else
                  dict(default=f.default, type=f.type))
        parser.add_argument(f'--{f.name}', **kwargs, **f.metadata)

    return parser
