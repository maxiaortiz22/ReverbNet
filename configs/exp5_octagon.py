"""v1 baseline (clean, LORO fold: octagon held out). See configs/v1_base.py."""
from configs.v1_base import make_config

globals().update(make_config(exp_num='5_octagon', add_noise=False, test_rooms=['octagon']))
