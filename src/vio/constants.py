"""Constants shared by the VIO backend wrappers."""

from __future__ import annotations

from pathlib import Path

#: standard gravity, used to convert the device's ``g`` accel units to ``m/s^2``
STD_GRAVITY = 9.80665

#: Keep the release version in sync with scripts/bootstrap.sh.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASALT_PREFIX = PROJECT_ROOT / ".deps" / "basalt" / "0.1.7"
DEFAULT_BASALT_BIN = BASALT_PREFIX / "bin" / "basalt_vio"
DEFAULT_BASALT_ETC = BASALT_PREFIX / "etc" / "basalt"

BASALT_INSTALL_HINT = f'bash "{PROJECT_ROOT / "scripts" / "bootstrap.sh"}"'
