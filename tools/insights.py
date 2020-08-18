#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
import math
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import slowfast.utils.misc as misc
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.checkpoint as cu
from slowfast.datasets import cv2_transform
from slowfast.visualization.utils import process_cv2_inputs
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
from slowfast.visualization.demo_loader import VideoReader
from slowfast.visualization.video_visualizer import VideoVisualizer
from visualization import visualize
from torchvision.utils import make_grid
from slowfast.models import build_model
from slowfast.visualization.utils import (
    GetWeightAndActivation,
    process_layer_index_data,
)
from slowfast.visualization.predictor import (
    ActionPredictor,
    Detectron2Predictor,
    draw_predictions,
)

logger = logging.get_logger(__name__)

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
            nrow = int(math.sqrt(reshaped_array.size()[0]))
            img_grid = make_grid(
                reshaped_array, nrow, padding=1, normalize=normalize
            )
            img_grid_res = img_grid.cpu().numpy().transpose(1, 2, 0)
            plt.title(name)
            plt.imshow(img_grid_res)
            plt.savefig(fname)


def prepare_inputs(task, cfg):
    frames, bboxes = task.frames, task.bboxes
    if bboxes is not None:
        bboxes = cv2_transform.scale_boxes(
            cfg.DATA.TEST_CROP_SIZE,
            bboxes,
            task.img_height,
            task.img_width,
        )
    if cfg.DEMO.INPUT_FORMAT == "BGR":
        frames = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames
        ]

    frames = [
        cv2_transform.scale(cfg.DATA.TEST_CROP_SIZE, frame)
        for frame in frames
    ]
    inputs = process_cv2_inputs(frames, cfg)
    if bboxes is not None:
        index_pad = torch.full(
            size=(bboxes.shape[0], 1),
            fill_value=float(0),
            device=bboxes.device,
        )
        # Pad frame index for each box.
        bboxes = torch.cat([index_pad, bboxes], axis=1)
    if cfg.NUM_GPUS > 0:
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
    return inputs, bboxes


def activations_to_frames(array):
    if array is not None and array.ndim != 0:
        if array.ndim == 1:
            reshaped_array = array.unsqueeze(0)
            nrow = int(math.sqrt(reshaped_array.size()[1]))
            reshaped_array = reshaped_array.view(-1, nrow)
            res = reshaped_array.cpu().numpy()
        elif array.ndim == 2:
            re = array.cpu().numpy()
        else:
            last2_dims = array.size()[-2:]
            reshaped_array = array.view(-1, *last2_dims)
            reshaped_array = reshaped_array.unsqueeze(1)
            nrow = int(math.sqrt(reshaped_array.size()[0]))
            img_grid = make_grid(
                reshaped_array, nrow, padding=1, normalize=normalize
            )
            res = img_grid.cpu().numpy().transpose(1, 2, 0)
        return res


def run_inf(frame_provider, cfg, model, object_detector, model_vis):
    video_vis = VideoVisualizer(
        cfg.MODEL.NUM_CLASSES,
        cfg.DEMO.LABEL_FILE_PATH,
        cfg.DETECTION.TOP_K,
        cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
        cfg.DETECTION.FADE,
        cfg.DETECTION.FONT_SIZE,
    )
    for able_to_read, task in frame_provider:
        if not able_to_read:
            break
        if cfg.DETECTION.ENABLE:
            task = object_detector(task)
        acti = None
        inputs, bboxes = prepare_inputs(task, cfg)
        activations, preds = model_vis.get_activations(inputs, bboxes)
        inputs, activations, preds = [inputs], [activations], [preds]
        import pdb; pdb.set_trace()
        acti = activations_to_frames(activations)
        frames = draw_predictions(task, video_vis, cfg.DETECTION.DRAW_RANGE)
        # hit Esc to quit the demo.
        key = cv2.waitKey(1)
        if key == 27:
            break
        yield frames, acti


def main():
    """
    Main function to launch the weight, activations and grad cam visualizations for a demo video.
    """
    args = parse_args()
    cfg = load_config(args)

    du.init_distributed_training(cfg)
    model = build_model(cfg)
    model.eval()
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)
    n_devices = cfg.NUM_GPUS * cfg.NUM_SHARDS

    logger.info("Start loading model weights")
    cu.load_test_checkpoint(cfg, model)
    logger.info("Finish loading model weights")

    prefix = "module/" if n_devices > 1 else ""
    # Get a list of selected layer names and indexing.
    layers = ["s1/pathway0_stem/conv"]
    layer_ls, indexing_dict = process_layer_index_data(
        layers, layer_name_prefix=prefix
    )
    model_vis = GetWeightAndActivation(model, layer_ls)

    logger.info("Start Model Visualization.")
    # if cfg.DEMO.MODEL_VIS.ENABLE and cfg.DEMO.MODEL_VIS.MODEL_WEIGHTS:
    #     logger.info("Start Weight Visualization.")
    #     # Register hooks for activations.
    #     layer_weights = model_vis.get_weights()

    #     for name, array in layer_weights.items():
    #         lw = array
    #         weights_to_image(lw, name, heat_map=True)
    #     logger.info("Finished Weight Visualization.")

    if cfg.DETECTION.ENABLE:
        object_detector = Detectron2Predictor(cfg)

    frame_provider = VideoReader(cfg)

    for frames, actis in run_inf(frame_provider, cfg, model, object_detector, model_vis):
        for frame, acti in zip(frames, actis):
            frame_provider.display(frame, acti)
        frame_provider.clean()



if __name__ == "__main__":
    main()
