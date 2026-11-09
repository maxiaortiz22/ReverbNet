"""Repeated training of exp6 on the same database, to measure run-to-run variation (TRAIN-09)."""
from configs.v1_base import make_config

globals().update(make_config(exp_num='6_r1', add_noise=True, train_replica=1))
