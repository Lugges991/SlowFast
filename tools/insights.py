#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
import math
import torch
import slowfast.utils.misc as misc
import slowfast.utils.logging as logging
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.distributed as du
from visualization import visualize
from torchvision.utils import make_grid
from slowfast.models import build_model
from slowfast.visualization.utils import (
    GetWeightAndActivation,
    process_layer_index_data,
)

import seaborn as sns
import matplotlib.pyplot as plt

def add_heatmap(tensor):
    """
    Add heatmap to 2D tensor.
    Args:
        tensor (tensor): a 2D tensor. Tensor value must be in [0..1] range.
    Returns:
        heatmap (tensor): a 3D tensor. Result of applying heatmap to the 2D tensor.
    """
    assert tensor.ndim == 2, "Only support 2D tensors."
    # Move tensor to cpu if necessary.
    if tensor.device != torch.device("cpu"):
        arr = tensor.cpu()
    else:
        arr = tensor
    arr = arr.numpy()
    # Get the color map by name.
    cm = plt.get_cmap("viridis")
    heatmap = cm(arr)
    heatmap = heatmap[:, :, :3]
    return torch.Tensor(heatmap).permute(2, 0, 1)

def weights_to_image(array, name, heat_map=False, path="./model_analysis", normalize=True):
    n = name.replace("/", "_")
    fname = path + "/" + n + "_weights" + ".png"

    if array is not None and array.ndim != 0:
        if array.ndim == 1:
            reshaped_array = array.unsqueeze(0)
            nrow = int(math.sqrt(reshaped_array.size()[1]))
            reshaped_array = reshaped_array.view(-1, nrow)
            if heat_map:
                reshaped_array = add_heatmap(reshaped_array)
                img_grid_res = reshaped_array.cpu().numpy()
                plt.title(name)
                plt.imshow(img_grid_res)
                plt.savefig(fname)
            else:
                img_grid_res = reshaped_array.cpu().numpy()
                plt.title(name)
                plt.imshow(img_grid_res)
                plt.savefig(fname)
        elif array.ndim == 2:
            reshaped_array = array
            if heat_map:
                heatmap = add_heatmap(reshaped_array)
                heatmap = heatmap.cpu().numpy()
                plt.title(name)
                plt.imshow(heatmap)
                plt.savefig(fname)
            else:
                reshaped_array = reshaped_array.cpu().numpy()
                plt.title(name)
                plt.imshow(reshaped_array)
                plt.savefig(fname)
        else:
            last2_dims = array.size()[-2:]
            reshaped_array = array.view(-1, *last2_dims)
            if heat_map:
                reshaped_array = [
                    add_heatmap(array_2d).unsqueeze(0)
                    for array_2d in reshaped_array
                ]
                reshaped_array = torch.cat(reshaped_array, dim=0)
            else:
                reshaped_array = reshaped_array.unsqueeze(1)
            import pdb;pdb.set_trace()
            nrow = int(math.sqrt(reshaped_array.size()[0]))
            img_grid = make_grid(
                reshaped_array, nrow, padding=1, normalize=normalize
            )
            img_grid_res = img_grid.cpu().numpy().transpose(1, 2, 0)
            plt.title(name)
            plt.imshow(img_grid_res)
            plt.savefig(fname)

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

    for name, array in layer_weights.items():
        lw = array
        weights_to_image(lw, name, heat_map=True)


if __name__ == "__main__":
    main()
