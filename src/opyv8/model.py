from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from numpy import ndarray


class PNodeArg(Protocol):
    @property
    def name(self) -> str:
        ...

    @property
    def shape(self) -> Any:
        ...

    @property
    def type(self) -> str:
        ...


class PSparseTensor(Protocol):
    values: ndarray
    indices: ndarray
    shape: tuple[int]

    @property
    def dtype(self) -> Any:
        ...


class PInferenceSession(Protocol):
    def run(
        self, output_names, input_feed: dict[str, Any], run_options=None
    ) -> list[ndarray] | list[list] | list[dict] | list[PSparseTensor]:
        ...

    def get_inputs(self) -> list[PNodeArg]:
        ...


@dataclass
class ImageTensor:
    original_size: tuple[int, int]
    scale_size: tuple[int, int]
    data: ndarray


@dataclass
class Label:
    x: int
    y: int
    width: int
    height: int
    classifier: str


@dataclass
class LabelImage:
    source: Optional[str]
    path: str
    width: int
    height: int
    labels: Sequence[Label]
