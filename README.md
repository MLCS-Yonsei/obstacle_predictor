# obstacle_predictor


## Requirements
https://github.com/rst-tu-dortmund/costmap_converter.git

## Launch
```bash
roslaunch obstacle_predictor obstacle_predictor.launch
```

## parameters
- ```global_frame_id```: The name of the coordinate frame published by the localization system. (default: ```/map```)
- ```base_frame_id```: Which frame to use for the robot base. (default: ```/base_footprint```)
- ```costmap_topic```: Topic name of the local costmap message, provided by the ```/move_base``` node. (default: ```/move_base/local_costmap/costmap```)
- ```obstacle_topic```: Topic name of the obstacle message to publish. (default: ```/move_base/TebLocalPlanner/obstacles```)
