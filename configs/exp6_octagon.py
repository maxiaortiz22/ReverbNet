"""v1 baseline (noisy, LORO fold: octagon held out). See configs/v1_base.py."""
from configs.v1_base import make_config

globals().update(make_config(exp_num='6_octagon', add_noise=True, test_rooms=['octagon']))
