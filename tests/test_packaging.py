from __future__ import annotations

import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_wheel_contains_apache_license(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            subprocess.run(
                ["uv", "build", "--wheel", "--out-dir", str(output)],
                cwd=REPOSITORY,
                check=True,
                capture_output=True,
                text=True,
            )
            [wheel] = output.glob("*.whl")
            with zipfile.ZipFile(wheel) as archive:
                license_paths = [
                    name for name in archive.namelist() if name.endswith(".dist-info/licenses/LICENSE")
                ]
                self.assertEqual(len(license_paths), 1)
                self.assertEqual(
                    archive.read(license_paths[0]),
                    (REPOSITORY / "LICENSE").read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
