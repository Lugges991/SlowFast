#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
import slowfast.utils.misc as misc
import slowfast.utils.logging as logging
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.distributed as du
from visualization import visualize
from slowfast.models import build_model
from slowfast.visualization.utils import (
    GetWeightAndActivation,
    process_layer_index_data,
)

import seaborn as sns
import matplotlib.pyplot as plt

def weights_to_image(weights, path="./weights.png")
    pass


logger = logging.get_logger(__name__)
def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    du.init_distributed_training(cfg)
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)
    n_devices = cfg.NUM_GPUS * cfg.NUM_SHARDS
    prefix = "module/" if n_devices > 1 else ""
    # Get a list of selected layer names and indexing.
    layers = ["s1/pathway0_stem/conv"]
    layer_ls, indexing_dict = process_layer_index_data(
        layers, layer_name_prefix=prefix
    )
    logger.info("Start Model Visualization.")
    # Register hooks for activations.
    model_vis = GetWeightAndActivation(model, layer_ls)
    layer_weights = model_vis.get_weights()
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
