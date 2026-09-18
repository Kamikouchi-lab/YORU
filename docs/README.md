# Overview
1. Creating a new project and creating a model.

    See [Create a model](training.md)

2. Evaluating a model.

    See [Evaluate models](evaluation.md)

3. Analyzing videos in offline.

    See [Analyze video](analysis.md)

4. Adopting a closed-loop system and performing real-time analysis.

    See [Closed-loop system](closed-loop.md)

# The YORU window

Every YORU screen opens sized to the monitor it is on and centred, so the same
project works on a laptop and on a large desktop display without anything being
cut off or left in an unreachable corner. The window can be resized or
maximised freely.

Each screen has a **Window** menu:

| Item | What it does |
|------|--------------|
| Fit window to this screen | Re-sizes and re-centres the window for the display it is on now. Use this after moving YORU to a different monitor. |
| Maximize window | Fills the screen. |
| Text size | Small / Normal / Large, applied immediately. |
| Save layout now | Writes the current arrangement out without waiting for you to close the window. |
| Reset layout to default | Forgets the saved arrangement and goes back to the built-in one. |

YORU remembers each screen's window size, text size, and — on the screens that
have more than one panel, Real-time Process and Video Analysis — where you put
the panels. The settings are per screen, so arranging the analysis window does
not disturb the training window.

On the two-panel screens the panels are tiled side by side and follow the
window as you resize it, until the first time you move or resize one yourself;
after that your arrangement is kept exactly as you left it. **Reset layout to
default** puts the tiling back.

These files live in `logs/` next to the project (or, if YORU is installed
somewhere it cannot write to, in `%LOCALAPPDATA%\YORU`): `yoru_windows.ini`
holds the window sizes and text sizes, and one `custom_layout_<screen>.ini` per
multi-panel screen holds the panel arrangement. Deleting them is the same as
choosing **Reset layout to default**.

Japanese (and other non-ASCII) text displays correctly throughout, including
file and folder names in the path fields.

# YORU documents
- [Home](../README.md)
- [Overview](README.md)
- [Install YORU](install.md)
- [Create a model](training.md)
- [Evaluate models](evaluation.md)
- [Analyze video](analysis.md)
- [Closed-loop system](closed-loop.md)
