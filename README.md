# ONNX-PREDICT-YOLOV8
This repository is a light weight library to ease the use of ONNX models exported by the Ultralytics YOLOv8 framework.

## Example Usage
```python
from onnxruntime import InferenceSession
from PIL import Image
from opyv8 import Predictor

model = Path("path/to/file.onnx")
# List of classes where the index match the class id in the ONNX network
classes = model.parent.joinpath("classes.names").read_text().split("\n")
session = InferenceSession(
    model.as_posix(),
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)
predictor = Predictor(session, classes)
img = Image.open("path/to/image.jpg")
print(predictor.predict(img))
```