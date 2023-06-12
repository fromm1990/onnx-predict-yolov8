from pathlib import Path
from typing import Optional

from onnxruntime import InferenceSession
from PIL import Image, ImageDraw, ImageOps
from PIL.Image import Image as PILImage
from typer import Option, Typer
from typing_extensions import Annotated

from opyv8 import Predictor
from opyv8.model import LabelImage

app = Typer()


def draw(image: PILImage, label_image: LabelImage) -> PILImage:
    image = ImageOps.exif_transpose(image)
    stage = ImageDraw.Draw(image)

    for label in label_image.labels:
        stage.rectangle(
            ((label.x, label.y), (label.x + label.width, label.y + label.height)),
            outline="magenta",
            width=2,
        )
    return image


@app.command()
def predict(
    model: Path, input: Path, class_file: Annotated[Optional[Path], Option()] = None
) -> None:
    model = Path(model)

    if not class_file:
        print(model.parent)
        class_file = model.parent.joinpath("classes.names")

    classes = class_file.read_text().split("\n")
    session = InferenceSession(
        model.as_posix(),
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    predictor = Predictor(session, classes)
    img = Image.open(input)
    img = draw(img, predictor.predict(img))

    out = input.parent.joinpath(f"annotated_{input.name}")
    img.save(out)
    # img.save(out)


if __name__ == "__main__":
    app()
