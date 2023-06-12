from __future__ import annotations

import numpy
from numpy import float32, float64, int32, int64
from numpy.typing import NDArray
from PIL import ImageOps
from PIL.Image import Image, Resampling

from opyv8.model import ImageTensor, PInferenceSession


def compute_iou(box: NDArray[int32], boxes: NDArray[int32]) -> NDArray[float64]:
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = numpy.minimum(box[0], boxes[:, 0])
    ymin = numpy.minimum(box[1], boxes[:, 1])
    xmax = numpy.maximum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
    ymax = numpy.maximum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])

    # Compute intersection area
    intersection_area = numpy.maximum(0, xmax - xmin) * numpy.maximum(0, ymax - ymin)

    # Compute union area
    box_area = box[2] * box[3]
    boxes_area = boxes[:, 2] * boxes[:, 3]
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def nms(
    boxes: NDArray[int32], scores: NDArray[float32], iou_threshold: float
) -> list[int64]:
    # Sort by score
    sorted_indices = numpy.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = numpy.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def image_to_tensor(img: Image, model: PInferenceSession) -> ImageTensor:
    _, _, width, height = model.get_inputs()[0].shape

    img = ImageOps.exif_transpose(img)
    original_size = img.size

    img = ImageOps.contain(img, (width, height), Resampling.BILINEAR)
    scale_size = img.size

    img = ImageOps.pad(
        img, (width, height), Resampling.BILINEAR, (114, 114, 114), (0, 0)
    )
    data = numpy.array(img)

    data = data / 255.0
    data = data.transpose(2, 0, 1)
    tensor = data[numpy.newaxis, :, :, :].astype(numpy.float32)

    return ImageTensor(original_size, scale_size, tensor)
