import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area
from typing import Any, Dict, List, Optional, Tuple
from .modeling.sam import Sam
from .predictor import SamPredictor
from .utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_points_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class SamAutomaticMaskGenerator:
    def __init__(
            self,
            model: Sam,
            points_per_side: Optional[int] = 32,
            points_per_batch: int = 64,
            pred_iou_thresh: float = 0.88,
            stability_score_thresh: float = 0.95,
            stability_score_offset: float = 1.0,
            box_nms_thresh: float = 0.7,
            crop_n_layers: int = 0,
            crop_nms_thresh: float = 0.7,
            crop_overlap_ratio: float = 512 / 1500,
            crop_n_points_downscale_factor: int = 1,
            point_grids: Optional[List[np.ndarray]] = None,
            min_mask_region_area: int = 0,
            output_mode: str = "binary_mask",
    ) -> None:
        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_points_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError(
                "Can't have both point_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coloc_rle":
            from pycocotools import mask as mask_utils

        if min_mask_region_area > 0:
            import cv2

        self.predicdtor = SamPredictor(model)
