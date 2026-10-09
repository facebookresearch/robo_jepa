# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Local-filesystem handler for the dataset loader.

Only local paths are supported. ``s3_profile`` is kept in the signature for config compatibility and must be
None; any other value raises.
"""

from fsspec.implementations.local import LocalFileSystem


class FileSystemHandler:
    def __init__(self, s3_profile=None):
        if s3_profile is not None:
            raise ValueError(f"S3 access is not supported here (got s3_profile={s3_profile!r}); use a local path.")
        self._fs = LocalFileSystem()

    def get_from_path(self, path):
        if str(path).startswith("s3://"):
            raise ValueError(f"S3 paths are not supported here: {path}")
        return self._fs

    def open(self, path, mode="rb"):
        return self.get_from_path(path).open(path, mode)
