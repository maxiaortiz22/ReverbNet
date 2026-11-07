"""v1 baseline (noisy, LORO fold: great_hall held out). See configs/v1_base.py."""
from configs.v1_base import make_config

globals().update(make_config(exp_num='6_great_hall', add_noise=True, test_rooms=['great_hall']))
