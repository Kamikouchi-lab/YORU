# Third-Party Licenses

This document lists all third-party dependencies used by YORU and their respective licenses.

## Direct Dependencies

| Package | Version | License | URL |
|---------|---------|---------|-----|
| eel | 0.16.0 | MIT | https://github.com/python-eel/Eel |
| nidaqmx | 0.9.0 | MIT | https://github.com/ni/nidaqmx-python |
| numpy | >=1.24.4 | BSD-3-Clause | https://www.numpy.org |
| dearpygui | 1.11.1 | MIT | https://github.com/hoffstadt/DearPyGui |
| opencv-python | 4.10.0.82 | Apache-2.0 | https://github.com/opencv/opencv-python |
| pyglet | 2.0.15 | BSD | https://pyglet.org |
| pyfirmata | 1.1.0 | MIT | https://github.com/tino/pyFirmata |
| pyyaml | 6.0.1 | MIT | https://pyyaml.org |
| pandas | 2.0.3 | BSD-3-Clause | https://pandas.pydata.org |
| pynput | 1.7.7 | LGPLv3 | https://github.com/moses-palmer/pynput |
| labelImg | >=1.8 | MIT | https://github.com/tzutalin/labelImg |
| munkres | 1.1.4 | Apache-2.0 | https://software.clapper.org/munkres/ |
| matplotlib | 3.7.5 | PSF | https://matplotlib.org |
| gitpython | 3.1.43 | BSD-3-Clause | https://github.com/gitpython-developers/GitPython |
| pillow | 10.3.0 | HPND | https://python-pillow.org |
| psutil | 5.9.8 | BSD-3-Clause | https://github.com/giampaolo/psutil |
| scipy | 1.10.1 | BSD-3-Clause | https://scipy.org |
| seaborn | 0.13.2 | BSD-3-Clause | https://github.com/mwaskom/seaborn |
| requests | 2.32.3 | Apache-2.0 | https://requests.readthedocs.io |
| thop | 0.1.1 | MIT | https://github.com/Lyken17/pytorch-OpCounter |
| tqdm | 4.66.4 | MPL-2.0 AND MIT | https://tqdm.github.io |
| ultralytics | >=8.3.0 | AGPL-3.0 | https://ultralytics.com |
| ultralytics-thop | 2.0.0 | AGPL-3.0 | https://github.com/ultralytics/thop |
| mss | 9.0.1 | MIT | https://github.com/BoboTiG/python-mss |
| scikit-image | 0.21.0 | BSD-3-Clause | https://scikit-image.org |
| imgui | 2.0.0 | BSD | https://github.com/pyimgui/pyimgui |
| pyopengl | 3.1.7 | BSD | http://pyopengl.sourceforge.net |
| pyopengl-accelerate | 3.1.7 | BSD | http://pyopengl.sourceforge.net |
| pyserial | 3.5 | BSD | https://github.com/pyserial/pyserial |
| pywin32 | 306 | PSF | https://github.com/mhammond/pywin32 |
| setuptools | <70 | MIT | https://github.com/pypa/setuptools |
| torch | >=2.0 | BSD-3-Clause | https://pytorch.org |
| torchvision | >=0.15 | BSD | https://github.com/pytorch/vision |

## Development Dependencies

| Package | Version | License | URL |
|---------|---------|---------|-----|
| pytest | 7.4.0 | MIT | https://docs.pytest.org |

## Notable Transitive Dependencies

| Package | License | URL |
|---------|---------|-----|
| bottle | MIT | http://bottlepy.org |
| bottle-websocket | MIT | https://github.com/bottlepy/bottle |
| certifi | MPL-2.0 | https://github.com/certifi/python-certifi |
| contourpy | BSD-3-Clause | https://github.com/contourpy/contourpy |
| filelock | Unlicense | https://github.com/tox-dev/filelock |
| gevent | MIT | https://www.gevent.org |
| greenlet | MIT | https://github.com/gevent/greenlet |
| imageio | BSD-2-Clause | https://github.com/imageio/imageio |
| jinja2 | BSD-3-Clause | https://jinja.palletsprojects.com |
| lxml | BSD-3-Clause | https://lxml.de |
| networkx | BSD-3-Clause | https://networkx.org |
| pyqt5 | GPL v3 | https://www.riverbankcomputing.com/software/pyqt/ |
| pyqt5-qt5 | LGPL v3 | https://www.riverbankcomputing.com/software/pyqt/ |
| pyqt5-sip | SIP License | https://www.riverbankcomputing.com/software/pyqt/ |
| polars | MIT | https://github.com/pola-rs/polars |
| sympy | BSD | https://www.sympy.org |

## Bundled Code

### labelImg

- **Location:** `yoru/labelimg/`
- **License:** MIT
- **Copyright:** (c) 2016 Tzutalin
- **License file:** `yoru/labelimg/LICENSE_labelImg`
- **Modifications:**
  - Default label format changed from PascalVOC to YOLO
  - Fixed classes.txt overwrite bug when loading YOLO annotations

## License Compatibility Notes

### Copyleft Licenses

1. **AGPL-3.0** (ultralytics, ultralytics-thop)
   - YORU itself is licensed under AGPL-3.0, so these are compatible.
   - Source code disclosure is required when distributing or serving over a network.

2. **GPL v3** (PyQt5)
   - PyQt5 is a transitive dependency via labelImg.
   - GPL v3 is compatible with AGPL-3.0 (AGPL is a superset).
   - Commercial PyQt5 licenses are available from Riverbank Computing.

3. **LGPLv3** (pynput)
   - Weak copyleft; compatible with AGPL-3.0.
   - Modifications to pynput itself must be disclosed.

### Ultralytics Commercial Licensing

Ultralytics YOLO is dual-licensed:

- **AGPL-3.0 (default):** Requires that any project incorporating Ultralytics code
  or models trained with it must be open-sourced under AGPL-3.0. This obligation
  extends to training code and models produced by that code. Modifications must be
  shared when the software is distributed or served over a network.

- **Enterprise License:** Removes the open-source requirement, permitting proprietary
  and private use. Allows commercial deployment and distribution without disclosing
  source code. Includes dedicated support and custom contractual terms.

**For commercial use:** An Enterprise license is required. Submit a request at
https://www.ultralytics.com/license for pricing and terms.

## License Abbreviations

| Abbreviation | Full Name |
|---|---|
| MIT | MIT License |
| BSD / BSD-3-Clause | BSD 3-Clause "New" or "Revised" License |
| BSD-2-Clause | BSD 2-Clause "Simplified" License |
| Apache-2.0 | Apache License 2.0 |
| AGPL-3.0 | GNU Affero General Public License v3.0 |
| GPL v3 | GNU General Public License v3.0 |
| LGPLv3 | GNU Lesser General Public License v3.0 |
| PSF | Python Software Foundation License |
| MPL-2.0 | Mozilla Public License 2.0 |
| HPND | Historical Permission Notice and Disclaimer |
