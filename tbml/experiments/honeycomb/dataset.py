import os
import threading

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from PIL import Image

from tbml.data import TarredImagesRandomAccessDataset


class LeJEPADataset:
    """Dataset that produces global and local views for LeJEPA training."""

    _dataset: TarredImagesRandomAccessDataset
    _resize_dim: int
    _num_global_views: int
    _global_view_dim: int
    _num_local_views: int
    _local_view_dim: int
    _rng: np.random.Generator
    _pid: int
    _rng_lock: threading.Lock
    _mean: np.ndarray | None
    _std: np.ndarray | None

    def __init__(
        self,
        dataset: TarredImagesRandomAccessDataset,
        resize_dim: int,
        num_global_views: int,
        global_view_dim: int,
        num_local_views: int,
        local_view_dim: int,
        mean_std: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    ) -> None:
        """Initialize the LeJEPA dataset wrapper.

        :param dataset: Base dataset providing PIL images.
        :param resize_dim: Shorter-side resize dimension.
        :param num_global_views: Number of global views per sample.
        :param global_view_dim: Crop size for global views.
        :param num_local_views: Number of local views per sample.
        :param local_view_dim: Crop size for local views.
        :param mean_std: Optional RGB mean/std for standardization.
        """
        if resize_dim <= 0:
            raise ValueError("resize_dim must be > 0")
        if num_global_views < 0:
            raise ValueError("num_global_views must be >= 0")
        if global_view_dim <= 0:
            raise ValueError("global_view_dim must be > 0")
        if num_local_views < 0:
            raise ValueError("num_local_views must be >= 0")
        if local_view_dim <= 0:
            raise ValueError("local_view_dim must be > 0")
        if num_global_views + num_local_views <= 0:
            raise ValueError("at least one view must be requested")
        if resize_dim < global_view_dim:
            raise ValueError("resize_dim must be >= global_view_dim")
        if resize_dim < local_view_dim:
            raise ValueError("resize_dim must be >= local_view_dim")

        self._dataset = dataset
        self._resize_dim = resize_dim
        self._num_global_views = num_global_views
        self._global_view_dim = global_view_dim
        self._num_local_views = num_local_views
        self._local_view_dim = local_view_dim
        self._rng = np.random.default_rng()
        self._pid = os.getpid()
        self._rng_lock = threading.Lock()
        if mean_std is None:
            self._mean = None
            self._std = None
        else:
            mean, std = mean_std
            if len(mean) != 3 or len(std) != 3:
                raise ValueError("mean_std must contain three values for mean and std")
            self._mean = np.asarray(mean, dtype=np.float32)
            self._std = np.asarray(std, dtype=np.float32)
            if np.any(self._std <= 0.0):
                raise ValueError("mean_std std values must be > 0")

    def __len__(self) -> int:
        """Return the number of samples in the base dataset.

        :returns: Dataset length.
        """
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Array:
        """Load and augment a sample into multiple views.

        :param idx: Dataset index.
        :returns: Array of shape (V_total, H_global, W_global, C).
        """
        if idx < 0 or idx >= len(self._dataset):
            raise IndexError("index out of range")

        current_pid = os.getpid()
        if current_pid != self._pid:
            self._rng = np.random.default_rng()
            self._pid = current_pid
            self._rng_lock = threading.Lock()

        image = self._dataset[idx]
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self._resize_short_side(image, self._resize_dim)

        global_views = [
            self._random_crop(image, self._global_view_dim) for _ in range(self._num_global_views)
        ]
        local_views = [
            self._random_crop(image, self._local_view_dim) for _ in range(self._num_local_views)
        ]
        local_views = [
            view.resize(
                (self._global_view_dim, self._global_view_dim),
                resample=Image.Resampling.BICUBIC,
            )
            for view in local_views
        ]

        views = global_views + local_views
        stacked = np.stack([np.asarray(view, dtype=np.float32) for view in views], axis=0)
        if self._mean is not None and self._std is not None:
            stacked = stacked / np.float32(255.0)
            stacked = (stacked - self._mean[None, None, None, :]) / self._std[None, None, None, :]
        return jnp.asarray(stacked)

    def _resize_short_side(self, image: Image.Image, target: int) -> Image.Image:
        """Resize the image so that the shorter side matches ``target``.

        :param image: Input image.
        :param target: Target size for the shorter side.
        :returns: Resized image.
        """
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("image must have positive dimensions")

        short_side = min(width, height)
        scale = target / float(short_side)
        new_width = max(int(round(width * scale)), 1)
        new_height = max(int(round(height * scale)), 1)
        if new_width == width and new_height == height:
            return image
        return image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)

    def _random_crop(self, image: Image.Image, crop_dim: int) -> Image.Image:
        """Sample a random square crop from the image.

        :param image: Input image.
        :param crop_dim: Side length of the square crop.
        :returns: Cropped image.
        """
        width, height = image.size
        if crop_dim > width or crop_dim > height:
            raise ValueError("crop_dim must not exceed resized image dimensions")

        max_left = width - crop_dim
        max_top = height - crop_dim
        with self._rng_lock:
            left = int(self._rng.integers(0, max_left + 1))
            top = int(self._rng.integers(0, max_top + 1))
        right = left + crop_dim
        bottom = top + crop_dim
        return image.crop((left, top, right, bottom))
