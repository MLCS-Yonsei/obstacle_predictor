# obstacle_predictor
ROS package which predicts the velocity of dynamic obstalces using local cost-map optical flow.


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
- ```prediction_horizon```: Time horizon for generating predicted obstacles. (default: ```1.0```)
- ```movement_tolerence```: If speed of some dynamic obstacles is slower than this parameter, those obstacles will be ignored. (default: ```0.1```)
- ```window_size```: Window size for Lucas-Kanade optical flow. (default: ```3```)
