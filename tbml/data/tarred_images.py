import io
import mmap
import os
import tarfile
import threading
from typing import ClassVar

from PIL import Image


class TarredImagesRandomAccessDataset:
    """Random-access dataset for images stored inside tar archives."""

    _IMAGE_EXTENSIONS: ClassVar[set[str]] = {
        ".bmp",
        ".gif",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }

    tar_paths: list[str]
    _file_handles: list[io.BufferedReader]
    _mmaps: list[mmap.mmap]
    _tars: list[tarfile.TarFile]
    _index: list[tuple[int, str]]
    _locks: list[threading.Lock]

    def __init__(self, tar_paths: list[str]) -> None:
        """Initialize the dataset and index tar members.

        :param tar_paths: List of tar file paths to index.
        """
        if len(tar_paths) == 0:
            raise ValueError("tar_paths must be non-empty")

        self.tar_paths = tar_paths
        self._file_handles = []
        self._mmaps = []
        self._tars = []
        self._index = []
        self._locks = []

        success = False
        try:
            for tar_idx, path in enumerate(tar_paths):
                if os.path.isfile(path) is False:
                    raise FileNotFoundError(f"tar file not found: {path}")

                file_handle = open(path, "rb")
                mapped = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)
                tar = tarfile.open(fileobj=mapped, mode="r:")

                self._file_handles.append(file_handle)
                self._mmaps.append(mapped)
                self._tars.append(tar)
                self._locks.append(threading.Lock())

                for member in tar.getmembers():
                    if member.isfile() is False:
                        continue
                    _, ext = os.path.splitext(member.name)
                    if ext.lower() not in self._IMAGE_EXTENSIONS:
                        continue
                    self._index.append((tar_idx, member.name))
            success = True
        finally:
            if success is False:
                self._close()

    def __len__(self) -> int:
        """Return the number of indexed images.

        :returns: Total number of images across all tar files.
        """
        return len(self._index)

    def __getitem__(self, idx: int) -> Image.Image:
        """Load an image by global index.

        :param idx: Global image index.
        :returns: PIL Image instance.
        :raises IndexError: If the index is out of range.
        :raises ValueError: If the tar member cannot be extracted as a file.
        """
        if idx < 0 or idx >= len(self._index):
            raise IndexError("index out of range")

        tar_idx, member_name = self._index[idx]
        tar = self._tars[tar_idx]
        with self._locks[tar_idx]:
            member_file = tar.extractfile(member_name)
            if member_file is None:
                raise ValueError(f"failed to extract member: {member_name}")

            try:
                payload = member_file.read()
            finally:
                member_file.close()

        image = Image.open(io.BytesIO(payload))
        image.load()
        return image

    def _close(self) -> None:
        for tar in self._tars:
            tar.close()
        for mapped in self._mmaps:
            mapped.close()
        for handle in self._file_handles:
            handle.close()
