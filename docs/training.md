# Creating models

1. Run the YORU's Training sub-module.

2. Create a project folder. (Step0)

    > Folders and condition yaml file will be created.

    > **Oriented Bounding Box (OBB)** — tick this box before creating the
    > project if your animals are elongated and lie at every angle (a fly, a
    > larva, a fish). The project is then labelled, trained and run with
    > *rotated* boxes instead of upright ones. See
    > [Oriented bounding boxes](#oriented-bounding-boxes) below.

    > The choice is written into the project's `config.yaml` as `task: obb`
    > and cannot be changed afterwards without relabelling, because the two
    > label formats are different files.

3. Extract frames for labeling using Grab GUI. (Step1)

   I. Select a video in the Video file path in the Grab GUI.

   Ⅱ. Select Save directory. (Basically, all_label_images in the project folder is a good choice.)

   Ⅲ. Decide the grabbed frame name. (Left blank, the video's own file name
   is used.)

   IV. Cut out the screenshot, by hand or automatically.

      i. Play video with Streaming movie.

      ii. Arrow keys to go forward and back.

      iii. Grab Current Frame or Alt key to save frame.

    ### Automatic Extraction

    Instead of looking for frames one at a time, **Automatic Extraction**
    picks a whole set at once, the way DeepLabCut's `extract_frames` does.
    Set *Frames to pick*, choose an algorithm, and press **Extract Frames**;
    the frames are written into the same folder, with the same names, as the
    ones grabbed by hand. **Stop** interrupts a run and keeps what it has
    already saved.

    > **uniform** draws frames at random from the range. It is instant, and
    > the sample mirrors how often each thing actually happens in the video.
    > It is the right choice when the behaviour you are labelling is common.

    > **kmeans** shrinks every frame to a thumbnail, clusters the thumbnails
    > by appearance and takes one frame per cluster. It has to read the range
    > once, so it takes a few seconds per thousand frames, but a rare posture
    > gets a cluster of its own instead of being swamped by the thousands of
    > near-identical frames of an animal sitting still. Prefer it when the
    > behaviour is rare, or when a uniform sample came back looking all the
    > same.

    > **Video range** is given as fractions of the video, so `0.25` to `0.75`
    > is the middle half. Use it to skip the handling at the start of a
    > recording, or to keep the end of a video back as unseen test material.

    > Extracting more than once adds new frames rather than replacing them:
    > each run draws independently, so a second run is a reasonable way to
    > enlarge a training set that turned out too small.

4. Run LabelImg and label the frames. (Step2)

    > YORU opens its own copy of LabelImg, already pointed at the project's
    > `all_label_images` folder and its `classes.txt`, and already in the
    > annotation format the project needs — YOLO for an ordinary project,
    > YOLO-OBB for an OBB one. You do not set the format by hand.

    > The general LabelImg documentation is at
    > [LabelImg](https://github.com/HumanSignal/labelImg); the two things YORU
    > adds are **Click to Box** and **oriented boxes**, described below.

    > It is easier to label if Auto Save mode is turned on in the View tab.

    ### Click to Box

    Instead of dragging a rectangle around each animal, press **C** (or the
    *Click to Box* button) and click once on the animal. A box is fitted to its
    body and given the current label.

    > The fit finds the animal's body axis and leaves its legs, wings and
    > antennae out of the box. In an OBB project the box comes out rotated
    > along the body; in an ordinary project you get the upright box around the
    > same fit.

    > It works on a dark animal on a light plate and on a pale animal on a dark
    > plate, deciding which from the pixels under the cursor. Nothing needs
    > installing and no model is needed — it runs on OpenCV alone.

    > The tool disarms itself as soon as a box appears, so your next click can
    > grab a corner to adjust it. Press **C** again for the next animal, or
    > **Escape** to cancel.

    > If it cannot find an animal where you clicked it says so in the status
    > bar and stays armed, so you can simply click again a little over. Drawing
    > the box by hand with **W** always works.

    ### Oriented bounding boxes

    In an OBB project a box can be turned to lie along the animal:

    | Key | Action |
    |---|---|
    | `Z` / `X` | turn the selected box 1° left / right |
    | `Shift+Z` / `Shift+X` | turn it 15° left / right |

    > Dragging a corner still resizes the box, and it stays square to its own
    > axes rather than to the image, so a tilted box is adjusted exactly like
    > an upright one.

    > The status bar shows the box's own width, height and angle, not those of
    > the upright box around it.

    > OBB labels are saved as `class x1 y1 x2 y2 x3 y3 x4 y4` (four corners,
    > normalised) — the format ultralytics reads for `task="obb"`. The file
    > extension is `.txt`, the same as ordinary YOLO labels, and LabelImg tells
    > the two apart by counting the numbers on a line.

5. Move all images and txt files to "all_label_images" folder of the project. (Step3)

6. Push "Move Label Images" button. (Step4)

    > Images and text files are copied to the train and val folders in a 4:1 ratio.

7. Select classes.txt file and push "Add class info in YAML file". (Step5)

    > The information in classes.txt will be entered into the config.yml file.

8. Check the "YAML Path" and select training conditions, such as epochs, networks and so on.

    > In an OBB project the weight gains an `-obb` suffix (`yolo11s-obb.pt`)
    > and the model family is fixed to YOLO: only YOLOv8 and YOLO11 have a
    > rotated-box head. RT-DETR, Faster R-CNN, Mask R-CNN and SSD cannot be
    > trained on oriented boxes.

    > The "GPU memory" line under the training conditions estimates how much
    > VRAM the run will need and compares it with what the card has free right
    > now. Green means it fits, orange means it fits with little headroom, and
    > red means the run is expected to hit a CUDA out-of-memory error.

    > The estimate is accurate to roughly ±30%, and it counts memory other
    > processes are already holding — so it drops if another training run or a
    > detection session is using the same card.

9. Start training by push "Train Model".

    >  In the terminal, you should check the initiation of training.

    > If the estimate says the run will not fit, YORU asks before starting and
    > offers the largest batch size it expects to fit.

10. To end a run early, push "Stop after this epoch".

    > Training keeps going until the epoch it is in has finished and its
    > checkpoint has been written, then ends the way a completed run does: for
    > YOLO and RT-DETR that includes the final validation pass, so `best.pt`
    > and `last.pt` in the run folder are both usable models.

    > How long this takes is one epoch at most, and the "Remaining" time is for
    > the whole run, not for the epoch. If that is still too long, "Force stop"
    > appears next to it and kills training immediately -- at the cost of the
    > epoch in progress and of the final validation.

    > The button writes an empty `.yoru_stop_request` file into the project
    > directory. A run started from a terminal can be stopped the same way by
    > creating that file by hand.

<img src="./imgs/screenshots_description_01.png" width="100%">

<img src="./imgs/screenshots_description-02.png" width="100%">

---

## What an OBB project changes, end to end

| Stage | Ordinary project | OBB project |
|---|---|---|
| `config.yaml` | `task: detect` | `task: obb` |
| LabelImg format | YOLO (`class cx cy w h`) | YOLO-OBB (`class x1 y1 … y4`) |
| Weight | `yolo11s.pt` | `yolo11s-obb.pt` |
| Model families | all of them | YOLOv8 / YOLO11 only |
| Real-time drawing | upright rectangle | rotated rectangle |
| `*_detect.csv` | `… total_time` | `… total_time, cx, cy, w, h, angle` |

The detection CSV gained five columns for **every** project, not only OBB ones:
`cx, cy, w, h` and `angle` (radians). For a model that does not predict
rotation these describe the same upright box as `x1..y2` with `angle = 0`, so
one reader handles both kinds of output. The original eight columns kept their
names and their positions, so existing analysis scripts and trigger plugins are
unaffected.

Analysis tables (`yoru.libs.analysis`) likewise gained `w`, `h` and `angle`
between `y_center` and `confidence`.

**One limitation to know about.** The evaluation sub-module
([Evaluate models](evaluation.md)) reads OBB label files, but computes IoU on
the *upright* box around each rotated one. Two boxes that overlap perfectly as
rectangles but differ in angle therefore score lower than they should, which
makes the mAP it reports for an OBB model conservative. The rotated mAP that
ultralytics prints at the end of training is the figure to quote.
