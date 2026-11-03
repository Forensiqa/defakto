# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Self

import numpy as np

from defakto.encoders import Encoder
from defakto.fitting.solver import FittingError, LabeledTrajectory

_EMBEDDINGS = "embeddings.f32"
_INDEX = "index.npz"
_META = "meta.json"


class TrajectoryStore:
    """Labeled embedding trajectories kept on disk, so fitting can read them several times.

    Encoding is the slow part of fitting, so it happens once: ``TrajectoryStore.encode``
    writes every trajectory to a folder, and fitting and calibration read it back from
    there without holding it all in memory.

    Embeddings are stored as float32, the precision the encoders compute in.
    ``meta.json`` is written last, so a folder without it is an unfinished store.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        meta_path = self.directory / _META
        if not meta_path.is_file():
            raise FittingError(
                f"'{self.directory}' is not a finished trajectory store (no {_META})"
            )
        self.meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        index = np.load(self.directory / _INDEX)
        self.offsets: np.ndarray = index["offsets"]
        self.labels: np.ndarray = index["labels"]
        self.dim = int(self.meta["dim"])
        total = int(self.offsets[-1])
        self._embeddings = (
            np.memmap(
                self.directory / _EMBEDDINGS, dtype=np.float32, mode="r", shape=(total, self.dim)
            )
            if total
            else np.zeros((0, self.dim), dtype=np.float32)
        )

    @classmethod
    def write(
        cls,
        directory: str | Path,
        trajectories: Iterable[LabeledTrajectory],
        *,
        meta: dict[str, Any] | None = None,
    ) -> Self:
        """Write labeled trajectories to ``directory`` and open the result."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / _META).unlink(missing_ok=True)

        offsets, labels, dim = [0], [], None
        with (directory / _EMBEDDINGS).open("wb") as f:
            for embeddings, positive in trajectories:
                e = np.asarray(embeddings, dtype=np.float32)
                if e.ndim != 2:
                    raise ValueError(f"trajectories must be 2-D (steps, dim), got shape {e.shape}")
                if dim is None:
                    dim = e.shape[1]
                elif e.shape[1] != dim:
                    raise FittingError(
                        f"trajectories disagree on the embedding size: {e.shape[1]} vs {dim}"
                    )
                f.write(np.ascontiguousarray(e).tobytes())
                offsets.append(offsets[-1] + len(e))
                labels.append(bool(positive))
        if dim is None:
            raise FittingError("no trajectories to store")

        np.savez(
            directory / _INDEX,
            offsets=np.asarray(offsets, dtype=np.int64),
            labels=np.asarray(labels, dtype=bool),
        )
        full_meta = {**(meta or {}), "dim": int(dim), "count": len(labels)}
        (directory / _META).write_text(json.dumps(full_meta, indent=2), encoding="utf-8")
        return cls(directory)

    @classmethod
    def encode(
        cls,
        directory: str | Path,
        encoder: Encoder,
        examples: Iterable[tuple[Any, bool]],
        *,
        meta: dict[str, Any] | None = None,
        on_progress: Callable[[int], None] | None = None,
    ) -> Self:
        """Encode ``(content, positive)`` examples and store their trajectories.

        ``on_progress`` is called with the number of examples encoded so far.
        """

        def trajectories() -> Iterator[LabeledTrajectory]:
            for i, (content, positive) in enumerate(examples, start=1):
                yield encoder.encode(content).embeddings, positive
                if on_progress is not None:
                    on_progress(i)

        return cls.write(
            directory, trajectories(), meta={"media_type": encoder.media_type, **(meta or {})}
        )

    @classmethod
    def open_if_current(cls, directory: str | Path, meta: dict[str, Any]) -> Self | None:
        """Open the store in ``directory`` if it is finished and was written with ``meta``.

        Returns None when there is no finished store there, or when any key of ``meta``
        differs, for example because it was encoded with another encoder.
        """
        try:
            store = cls(directory)
        except FittingError:
            return None
        if any(store.meta.get(key) != value for key, value in meta.items()):
            return None
        return store

    def __len__(self) -> int:
        return len(self.labels)

    def __iter__(self) -> Iterator[LabeledTrajectory]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i: int) -> LabeledTrajectory:
        start, end = int(self.offsets[i]), int(self.offsets[i + 1])
        return np.asarray(self._embeddings[start:end]), bool(self.labels[i])

    @property
    def lengths(self) -> np.ndarray:
        """Number of steps of each trajectory."""
        return np.diff(self.offsets)

    @property
    def num_positive(self) -> int:
        return int(self.labels.sum())

    @property
    def num_negative(self) -> int:
        return len(self) - self.num_positive
