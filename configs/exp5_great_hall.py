"""v1 baseline (clean, LORO fold: great_hall held out). See configs/v1_base.py."""
from configs.v1_base import make_config

globals().update(make_config(exp_num='5_great_hall', add_noise=False, test_rooms=['great_hall']))
