"""v1 baseline (noisy, dev split: classroom held out). See configs/v1_base.py."""
from configs.v1_base import make_config

globals().update(make_config(exp_num=6, add_noise=True))
