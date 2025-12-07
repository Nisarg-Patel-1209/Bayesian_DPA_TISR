# Copyright (c) OpenMMLab. All rights reserved.

try:
    import mmcv  # optional
except Exception:
    mmcv = None

from .version import __version__, version_info

__all__ = ["__version__", "version_info"]
