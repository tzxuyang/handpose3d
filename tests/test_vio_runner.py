from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import SRC_DIR  # noqa: F401
from vio import runner


class ProjectLocalBasaltTests(unittest.TestCase):
    def test_relative_binary_survives_subprocess_working_directory_change(self):
        with tempfile.TemporaryDirectory(prefix="basalt runner ") as tmp:
            root = Path(tmp)
            install = root / ".deps/basalt/0.1.7"
            binary = install / "bin/basalt_vio"
            binary.parent.mkdir(parents=True)
            (install / "lib").mkdir()
            binary.write_text(
                "#!/bin/sh\n"
                "printf '1.0 0 0 0 0 0 0 1\\n' > trajectory.txt\n"
                "printf '%s\\n' \"$LD_LIBRARY_PATH\"\n"
                "printf '%s\\n' \"$@\"\n"
            )
            binary.chmod(0o755)
            output = root / "output directory"
            for inherited in ("", "/existing/libraries"):
                with self.subTest(inherited=inherited), \
                        patch.object(runner, "PROJECT_ROOT", root), \
                        patch.dict(os.environ, {"LD_LIBRARY_PATH": inherited}):
                    trajectory = runner.run_basalt_vio(
                        root / "dataset", root / "calib.json", root / "config.json",
                        output, binary="./.deps/basalt/0.1.7/bin/basalt_vio",
                        keep_marg_data=True, verbose=False,
                    )
                    self.assertEqual(trajectory, output / "trajectory.txt")
                    self.assertTrue(trajectory.is_file())
                    lines = (output / "basalt_vio.log").read_text().splitlines()
                    expected_lib = str(install / "lib")
                    if inherited:
                        expected_lib += ":" + inherited
                    self.assertEqual(lines[0], expected_lib)
                    self.assertEqual(lines[-2:], ["--marg-data", str(output / "marg_data")])
                    self.assertEqual(os.environ["LD_LIBRARY_PATH"], inherited)


if __name__ == "__main__":
    unittest.main()
