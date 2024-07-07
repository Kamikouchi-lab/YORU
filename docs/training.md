# Creating models

1. Run the YORU's Training sub-module.

2. Create a project folder. (Step0)
    - Folders and condition yaml file will be created.

3. Extract frames for labeling using Grab GUI. (Step1)

   I. Select a video in the Video file path in the Grab GUI.

   Ⅱ. Select Save directory. (Basically, all_label_images in the project folder is a good choice.)

   Ⅲ. Decide the grabed frame name.

   IV. Cut out the screenshot.

      i. Play video with Streaming movie.

      ii. Arrow keys to go forward and back.

      iii. Grab Current Frame or Alt key to save frame.

4. Run LabelImg and label the frames. (Step2)

    > The detailed coduments is accessed in [LabelImg](https://github.com/HumanSignal/labelImg).

    > Save format is done in YOLO. 

    > It is easier to do so if Auto Save mode is turned on in the View tab.

5. Move all images and txt files to "all_label_images" folder of the project. (Step3)

6. Push "Move Label Images" button. (Step4)

    > Images and text files are copied to the train and val folders in a 4:1 ratio.

7. Select classes.txt file and push "Add class info in YAML file". (Step5)

    > The information in classes.txt will be entered into the config.yml file.

8. Check the "YAML Path" and select training conditions, such as epochs, networks and so on.

9. Start training by push "Train YOLOv5".

    >  In the terminal, you should check the initiation of training.
