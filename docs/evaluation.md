# Model evaluation

1. Run the YORU's Evaluation sub-module.

2. Load a project config.yaml file and a model.
    
    > The model is in the "exp" folder.

3. Extract frames for labeling using Grab GUI. 

   I. Select a video in the Video file path in the Grab GUI.

   Ⅱ. Select Save directory. (Basically, all_label_images in the project folder is a good choice.)

   Ⅲ. Decide the grabed frame name.

   IV. Cut out the screenshot.

      i. Play video with Streaming movie.

      ii. Arrow keys to go forward and back.

      iii. Grab Current Frame or Alt key to save frame.

   > Images that are not used for creating a model are better.

5. Run LabelImg and label the frames.

    > The detailed documents are accessible in [LabelImg](https://github.com/HumanSignal/labelImg).

    > Save format is done in YOLO. 

    > It is easier to do so if Auto Save mode is turned on in the View tab.

6. Push "Prediction" button.

7. Push "Calculate APs" button. 

    > YORU calculates APs and IOUs.
