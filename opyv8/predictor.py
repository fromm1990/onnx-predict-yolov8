from __future__ import annotations

from pathlib import Path
from typing import List, cast

import numpy
import utils
from model import Label, LabelImage, PInferenceSession
from numpy import ndarray
from PIL import Image as ImageModule
from PIL.Image import Image


class Predictor:
    def __init__(
        self,
        model: PInferenceSession,
        names: list[str],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
    ) -> None:
        self.__model = model
        self.__names = names
        self.__conf_threshold = conf_threshold
        self.__iou_threshold = iou_threshold

    def predict(self, img: Image | Path | str) -> LabelImage:
        if isinstance(img, str):
            img = Path(img)
        if isinstance(img, Path):
            img = ImageModule.open(img)
        if not isinstance(img, Image):
            raise ValueError("img must be an PIL Image, Path or string")

        tensor = utils.image_to_tensor(img, self.__model)
        results = cast(List[ndarray], self.__model.run(None, {"images": tensor.data}))
        predictions = numpy.squeeze(results[0]).T

        scores = numpy.max(predictions[:, 4:], axis=1)
        keep = scores > self.__conf_threshold
        predictions = predictions[keep, :]
        scores = scores[keep]
        class_ids = numpy.argmax(predictions[:, 4:], axis=1)

        boxes = predictions[:, :4]
        # Make x0, y0 left upper corner instead of box center
        boxes[:, 0:2] -= boxes[:, 2:4] / 2
        boxes /= numpy.array(
            [*tensor.scale_size, *tensor.scale_size], dtype=numpy.float32
        )
        boxes *= numpy.array([*tensor.original_size, *tensor.original_size])
        boxes = boxes.astype(numpy.int32)

        keep = utils.nms(boxes, scores, self.__iou_threshold)
        labels = []
        for bbox, label in zip(boxes[keep], class_ids[keep]):
            labels.append(
                Label(
                    x=bbox[0].item(),
                    y=bbox[1].item(),
                    width=bbox[2].item(),
                    height=bbox[3].item(),
                    classifier=self.__names[label],
                )
            )

        img_width, img_height = img.size
        return LabelImage(
            source=None,
            path=img.filename,  # type: ignore
            width=img_width,
            height=img_height,
            labels=labels,
        )
