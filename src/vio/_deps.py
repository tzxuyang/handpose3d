"""Import shim for the parent ``src`` modules.

The repository is run in two styles: entry points put ``src`` on ``sys.path``
(``import mcap_utils``) while some tooling imports the package form
(``from src.mcap_utils import ...``).  Centralising the fallback here keeps the
``vio`` modules readable.
"""

from __future__ import annotations

try:  # `src` is on sys.path (main.py, scripts, tests)
    from mcap_utils import extract_h264_data, read_hand_json_3d, read_mcap_json, read_mcap_protobuf, write_3d_hand_keypoints_mcap
except ImportError:  # imported as `src.vio.*`
    from src.mcap_utils import (  # type: ignore
        extract_h264_data,
        read_hand_json_3d,
        read_mcap_json,
        read_mcap_protobuf,
        write_3d_hand_keypoints_mcap,
    )

__all__ = [
    "extract_h264_data",
    "read_hand_json_3d",
    "read_mcap_json",
    "read_mcap_protobuf",
    "write_3d_hand_keypoints_mcap",
]
