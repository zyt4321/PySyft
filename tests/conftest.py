# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path

"""
    Dummy conftest.py for syft.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""

# Make sure that the application source directory (this directory's parent) is
# on sys.path.

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
syft_src_path = Path(root_path) / "src"
sys.path.append(str(syft_src_path))
