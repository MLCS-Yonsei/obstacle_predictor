#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import PolygonStamped, Point32
try:
    from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
except:
    rospy.logerr('Failed to import ObstacleArrayMsg, ObstacleMsg.')

from cv2 import resize, calcOpticalFlowFarneback, GaussianBlur
from scipy.ndimage.filters import gaussian_filter, median_filter

from utils import *
import time



class ObstaclePredictor:

    def __init__(self):

        # ROS parameters
        self.global_frame = rospy.get_param("/obstacle_predictor/global_frame_id")
        self.global_frame = self.global_frame[1:] \
            if self.global_frame[0]=='/' else self.global_frame
        self.base_frame = rospy.get_param("/obstacle_predictor/base_frame_id")
        self.global_costmap_topic = rospy.get_param("/obstacle_predictor/global_costmap_topic")
        self.local_costmap_topic = rospy.get_param("/obstacle_predictor/local_costmap_topic")
        self.obstacle_topic = rospy.get_param("/obstacle_predictor/obstacle_topic")
        self.movement_tol_max = rospy.get_param("/obstacle_predictor/movement_tol_max")
        self.movement_tol_min = rospy.get_param("/obstacle_predictor/movement_tol_min")
        self.prediction_horizon = rospy.get_param("/obstacle_predictor/prediction_horizon")
        self.timediff_tol = rospy.get_param("/obstacle_predictor/timediff_tol")
        self.flowdiff_tol = rospy.get_param("/obstacle_predictor/flowdiff_tol")
        self.window_size = rospy.get_param("/obstacle_predictor/window_size")

        # Initialize ros node
        rospy.init_node('obstacle_predictor', anonymous=True)

        # Costmap buffer
        self.global_costmap_msg = None
        self.local_costmap_msg = None
        self.prev_local_costmap_msg = None

        # Publisher and subscriber
        self.global_costmap_sub = rospy.Subscriber(self.global_costmap_topic, OccupancyGrid, self.globalCostmapCallback)
        self.local_costmap_sub = rospy.Subscriber(self.local_costmap_topic, OccupancyGrid, self.localCostmapCallback)
        self.global_costmap_update_sub = rospy.Subscriber(
            self.global_costmap_topic+'_updates',
            OccupancyGridUpdate,
            self.globalCostmapUpdateCallback
        )
        self.local_costmap_update_sub = rospy.Subscriber(
            self.local_costmap_topic+'_updates',
            OccupancyGridUpdate,
            self.localCostmapUpdateCallback
        )
        self.obstacle_pub = rospy.Publisher(self.obstacle_topic, ObstacleArrayMsg, queue_size=10)


    def spin(self):
        rospy.spin()


    def globalCostmapCallback(self, msg):
        '''
        Save current global costmap to buffer
        '''
        self.global_costmap_msg = reshapeCostmap(msg)


    def localCostmapCallback(self, msg):
        '''
        Save current local costmap to buffer
        '''
        if isOccupancyGrid(self.global_costmap_msg):
            self.local_costmap_msg = reshapeCostmap(msg)
            self.mask_costmap(self.local_costmap_msg)
            if not isOccupancyGrid(self.prev_local_costmap_msg):
                self.prev_local_costmap_msg = copy(self.local_costmap_msg)
            self.predict_velocities()


    def globalCostmapUpdateCallback(self, msg):
        '''
        Update global costmap buffer
        '''
        if isOccupancyGrid(self.global_costmap_msg):
            updateCostmap(self.global_costmap_msg, msg)


    def localCostmapUpdateCallback(self, msg):
        '''
        Update local costmap buffer
        '''
        if isOccupancyGrid(self.local_costmap_msg) and isOccupancyGrid(self.global_costmap_msg):
            updateCostmap(self.local_costmap_msg, msg)
            self.mask_costmap(self.local_costmap_msg)
            self.predict_velocities()


    def predict_velocities(self):
        '''
        Compute optical flow of local costmap to predict velocities of moving obstacles.
        Predicted velocities will be used for generating ObstacleArrayMsg for TebLocalPlannerROS.
        '''
        if isOccupancyGrid(self.prev_local_costmap_msg):
            # Compute opticalFlowLK here.
            dt = self.local_costmap_msg.header.stamp.to_sec() - self.prev_local_costmap_msg.header.stamp.to_sec()
            if dt >0.01 and dt < self.timediff_tol: # skip opticalflow when dt is larger than self.timediff_tol (sec).
                I1g, I2g = self.preprocess_images()
                # flow, rep_physics = opticalFlowLK(I2g, I1g, self.window_size)
                # flow = -flow
                flow = -calcOpticalFlowFarneback(I2g, I1g, None, 0.5, 3, self.window_size, 3, 5, 1.2, 0)
                flow_ = calcOpticalFlowFarneback(I1g, I2g, None, 0.5, 3, self.window_size, 3, 5, 1.2, 0)
                flowdiff = np.linalg.norm(flow - flow_, axis=2) > self.flowdiff_tol
                flow[:,:,0][flowdiff] = 0
                flow[:,:,1][flowdiff] = 0
                # flow[:,:,0] = gaussian_filter(flow[:,:,0], 3.0)
                # flow[:,:,1] = gaussian_filter(flow[:,:,1], 3.0)
                flow[:,:,0] = median_filter(flow[:,:,0], self.window_size)
                flow[:,:,1] = median_filter(flow[:,:,1], self.window_size)
        
                # Generate and Publish ObstacleArrayMsg
                self.publish_obstacles(flow, dt)

                # Save current costmap to buffer
                self.prev_local_costmap_msg = copy(self.local_costmap_msg)


    def publish_obstacles(self, flow, dt):
        '''
        Generate and publish ObstacleArrayMsg from flow vectors.
        '''
        obstacle_vels = np.transpose(flow, axes=[1,0,2]) / dt * self.local_costmap_msg.info.resolution # opt_uv needs to be transposed. ( [y,x] -> [x,y] )
        mask_img = gaussian_filter(self.local_costmap_msg.data.T, 1.0)

        # Generate obstacle_msg for TebLocalPlanner here.
        obstacle_msg = ObstacleArrayMsg()
        obstacle_msg.header.stamp = rospy.Time.now()
        obstacle_msg.header.frame_id = self.global_frame
        robot_pose = (
            self.local_costmap_msg.info.origin.position.x + self.local_costmap_msg.info.resolution *self.local_costmap_msg.info.height/2.0,
            self.local_costmap_msg.info.origin.position.y + self.local_costmap_msg.info.resolution *self.local_costmap_msg.info.width/2.0
        )
        obstacle_speed = np.linalg.norm(obstacle_vels, axis=2)
        obstacle_speed[mask_img < 40] = 0
        for i in range(obstacle_vels.shape[0]):
            for j in range(obstacle_vels.shape[1]):
                # Add point obstacle to the message if obstacle speed is larger than self.movement_tol.
                if obstacle_speed[i, j] > self.movement_tol_min and obstacle_speed[i, j] < self.movement_tol_max:
                    flow_vector_position = (
                        self.local_costmap_msg.info.origin.position.x + self.local_costmap_msg.info.resolution*(i+0.5),
                        self.local_costmap_msg.info.origin.position.y + self.local_costmap_msg.info.resolution*(j+0.5)
                    )
                    line_scale = (
                        1. - np.exp(
                            -np.linalg.norm([
                                flow_vector_position[0] - robot_pose[0],
                                flow_vector_position[1] - robot_pose[1]
                            ])/2.
                        )
                    )* self.prediction_horizon
                    obstacle_msg.obstacles.append(ObstacleMsg())
                    obstacle_msg.obstacles[-1].id = len(obstacle_msg.obstacles)-1
                    obstacle_msg.obstacles[-1].polygon.points = [Point32(), Point32()]
                    obstacle_msg.obstacles[-1].polygon.points[0].x = flow_vector_position[0]
                    obstacle_msg.obstacles[-1].polygon.points[0].y = flow_vector_position[1]
                    obstacle_msg.obstacles[-1].polygon.points[0].z = 0
                    obstacle_msg.obstacles[-1].polygon.points[1].x = flow_vector_position[0] + obstacle_vels[i, j, 0]*line_scale
                    obstacle_msg.obstacles[-1].polygon.points[1].y = flow_vector_position[1] + obstacle_vels[i, j, 1]*line_scale
                    obstacle_msg.obstacles[-1].polygon.points[1].z = 0

        self.obstacle_pub.publish(obstacle_msg)     # Publish predicted obstacles


    def mask_costmap(self, costmap_msg):
        '''
        Resize costmap data to 2D array and mask static obstacles.
        '''
        dx = costmap_msg.info.origin.position.x - self.global_costmap_msg.info.origin.position.x
        dy = costmap_msg.info.origin.position.y - self.global_costmap_msg.info.origin.position.y

        multiplier = float(costmap_msg.info.resolution)/float(self.global_costmap_msg.info.resolution)

        di = int(round(dy/self.global_costmap_msg.info.resolution))
        dj = int(round(dx/self.global_costmap_msg.info.resolution))

        mask = resize(
            self.global_costmap_msg.data[
                di:di+int(round(costmap_msg.info.height*multiplier)),
                dj:dj+int(round(costmap_msg.info.width*multiplier))
            ],
            dsize=(costmap_msg.info.height, costmap_msg.info.width)
        )
        costmap_msg.data[mask > 0] = 0


    def preprocess_images(self):
        '''
        Preprocess images for optical flow.
        Match the location of current costmap and previous costmap.
        '''
        w = self.local_costmap_msg.info.width
        h = self.local_costmap_msg.info.height
        dx = self.local_costmap_msg.info.origin.position.x - self.prev_local_costmap_msg.info.origin.position.x
        dy = self.local_costmap_msg.info.origin.position.y - self.prev_local_costmap_msg.info.origin.position.y

        di = int(round(dy/self.local_costmap_msg.info.resolution))
        dj = int(round(dx/self.local_costmap_msg.info.resolution))

        img_prev = np.zeros_like(self.prev_local_costmap_msg.data)
        img_curr = np.zeros_like(self.local_costmap_msg.data)
        if di < 0:
            if dj < 0:
                img_prev[-di:h, -dj:w] = self.prev_local_costmap_msg.data[:h+di, :w+dj]
                img_curr[-di:h, -dj:w] = self.local_costmap_msg.data[-di:h, -dj:w]
            else:
                img_prev[-di:h, :w-dj] = self.prev_local_costmap_msg.data[:h+di, dj:w]
                img_curr[-di:h, :w-dj] = self.local_costmap_msg.data[-di:h, :w-dj]
        else:
            if dj < 0:
                img_prev[:h-di, -dj:w] = self.prev_local_costmap_msg.data[di:h, :w+dj]
                img_curr[:h-di, -dj:w] = self.local_costmap_msg.data[:h-di, -dj:w]
            else:
                img_prev[:h-di, :w-dj] = self.prev_local_costmap_msg.data[di:h, dj:w]
                img_curr[:h-di, :w-dj] = self.local_costmap_msg.data[:h-di, :w-dj]
        
        return GaussianBlur(img_prev,(5,5),3), GaussianBlur(img_curr,(5,5),3)

##  End of class ObstaclePredictor.



if __name__ == '__main__':

    try:
        node = ObstaclePredictor()
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("node terminated.")

