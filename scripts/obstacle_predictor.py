#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import copy
from scipy import signal
from scipy.ndimage.filters import gaussian_filter

import rospy
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from geometry_msgs.msg import PolygonStamped, Point32
try:
    from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
except:
    rospy.logerr('Failed to import ObstacleArrayMsg, ObstacleMsg.')

import numpy as np
import cv2



class ObstaclePredictor:

    def __init__(self):

        # ROS parameters
        self.global_frame = rospy.get_param("/obstacle_predictor/global_frame_id")
        self.base_frame = rospy.get_param("/obstacle_predictor/base_frame_id")
        self.global_costmap_topic = rospy.get_param("/obstacle_predictor/global_costmap_topic")
        self.local_costmap_topic = rospy.get_param("/obstacle_predictor/local_costmap_topic")
        self.obstacle_topic = rospy.get_param("/obstacle_predictor/obstacle_topic")
        self.prediction_horizon = rospy.get_param("/obstacle_predictor/prediction_horizon")
        self.movement_tol = rospy.get_param("/obstacle_predictor/movement_tolerence")
        self.timediff_tol = rospy.get_param("/obstacle_predictor/timediff_tolerence")
        self.window_size = rospy.get_param("/obstacle_predictor/window_size")

        # Initialize ros node
        rospy.init_node('obstacle_predictor', anonymous=True)

        # Costmap buffer
        self.global_costmap_msg = None
        self.local_costmap_msg = None
        self.prev_DBSCAN_result = None

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
        Predict velocities of moving obstacles.
        Predicted velocities will be used for generating ObstacleArrayMsg for TebLocalPlannerROS.
        '''
        data = None # TODO
        db = DBSCAN(eps=0.3, min_samples=10).fit(data)
        if type(self.prev_DBSCAN_result) != type(None)
            # TODO
            line_start_x = None
            line_start_y = None
            line_end_x   = None
            line_end_y   = None
            # Generate and Publish ObstacleArrayMsg
            self.publish_obstacles(line_start_x, line_start_y, line_end_x, line_end_y)
            
        # Save current DBSCAN result to buffer
        self.prev_DBSCAN_result = db # TODO


    def publish_obstacles(self, line_start_x, line_start_y, line_end_x, line_end_y):
        '''
        Generate and publish ObstacleArrayMsg.
        '''
        obstacle_msg = ObstacleArrayMsg()
        obstacle_msg.header.stamp = rospy.Time.now()
        obstacle_msg.header.frame_id = self.global_frame[1:] \
            if self.global_frame[0]=='/' else self.global_frame
        obstacle_msg.obstacles.append(ObstacleMsg())
        obstacle_msg.obstacles[-1].id = 0
        obstacle_msg.obstacles[-1].polygon.points = [Point32(), Point32()]
        obstacle_msg.obstacles[-1].polygon.points[0].x = line_start_x
        obstacle_msg.obstacles[-1].polygon.points[0].y = line_start_y
        obstacle_msg.obstacles[-1].polygon.points[0].z = 0
        obstacle_msg.obstacles[-1].polygon.points[1].x = line_end_x
        obstacle_msg.obstacles[-1].polygon.points[1].y = line_end_y
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

        mask = cv2.resize(
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
        
        return img_prev, img_curr

##  End of class ObstaclePredictor.



def isOccupancyGrid(msg):
    '''
    Return False if msg is not an OccupancyGrid class.
    '''
    return type(msg) == type(OccupancyGrid())


def isOccupancyGridUpdate(msg):
    '''
    Return False if msg is not an OccupancyGridUpdate class.
    '''
    return type(msg) == type(OccupancyGridUpdate())


def reshapeCostmap(msg):
    '''
    Reshape, remove negative values and change type as uint8 (for cv2.resize)
    '''
    msg.data = np.reshape(
        msg.data,
        [msg.info.height, msg.info.width]
    ).clip(0).astype(np.uint8)
    return msg


def updateCostmap(costmap_msg, update_msg):
    temp = copy(costmap_msg)
    temp.header.stamp = update_msg.header.stamp
    temp.info.width = update_msg.width
    temp.info.height = update_msg.height
    temp.data = update_msg.data
    costmap_msg = reshapeCostmap(temp)


if __name__ == '__main__':

    try:
        node = ObstaclePredictor()
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("node terminated.")

