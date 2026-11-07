"""v1 baseline (clean, dev split: classroom held out). See configs/v1_base.py."""
from configs.v1_base import make_config

globals().update(make_config(exp_num=5, add_noise=False))
