## Data description
For test data, we can create and analyze fly mating behavior.

In this data, we labeled copulation paired flies and non copulation flies.

- labeled frames in [labeled_frames](labeled_frames) folder (1000 PNG images and 1000 labels text files)

## Demonstration steps
### Create a model

1. Download test_data.
  
2. Run "Training".

3. Create a project.

4. Move labeled data and classes.txt to "all_label_images" folder in the project.

5. Push "Move Label Images" button.

6. Select classes.txt files path and push "Add class info in YAML file".

7. Select training conditions and start training.

### Analyze a model

1. Select "Model Path", "Movie Path",and "Result Directory". "test_video_fly_copulation.mp4" is used as a movie.

2. Start analysis.


## Data references

These data ware used in the previous paper.

- [Paper](https://ieeexplore.ieee.org/document/10150245)

- H. M. Yamanouchi, R. Tanaka and A. Kamikouchi, "Event-triggered feedback system using YOLO for optogenetic manipulation of neural activity," 2023 IEEE International Conference on Pervasive Computing and Communications Workshops and other Affiliated Events (PerCom Workshops), Atlanta, GA, USA, 2023, pp. 184-187, doi: 10.1109/PerComWorkshops56833.2023.10150245.

