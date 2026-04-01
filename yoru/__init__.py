# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import os
import sys

__version__ = "1.1.0"  # ←リリース時にここだけ更新

# Ensure the project root is on sys.path for direct script execution
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
