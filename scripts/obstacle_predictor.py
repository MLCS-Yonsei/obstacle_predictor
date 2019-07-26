#!/usr/bin/env python

from copy import copy

import rospy
from nav_msgs.msg import OccupancyGrid
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
        self.costmap_topic = rospy.get_param("/obstacle_predictor/costmap_topic")
        self.obstacle_topic = rospy.get_param("/obstacle_predictor/obstacle_topic")
        teststring = self.global_frame[1:] \
            if self.global_frame[0]=='/' else self.global_frame
        print 'Test:', teststring

        # Initialize ros node
        rospy.init_node('obstacle_predictor', anonymous=True)

        # Costmap buffer
        self.prev_costmap_msg = OccupancyGrid()

        # Publisher and subscriber
        self.costmap_sub = rospy.Subscriber(self.costmap_topic, OccupancyGrid, self.costmapCallback)
        self.obstacle_pub = rospy.Publisher(self.obstacle_topic, ObstacleArrayMsg, queue_size=10)


    def spin(self, rate=None):
        if type(rate)==type(None):
            rospy.spin()
        else:
            self.rate = rospy.Rate(rate)
            while not rospy.is_shutdown():
                costmap_msg = rospy.wait_for_message(costmap_topic, OccupancyGrid)
                self.costmapCallback(costmap_msg)
                self.rate.sleep()


    def costmapCallback(self, costmap_msg):

        self.publish_obstacles(costmap_msg)
        self.prev_costmap_msg = copy(costmap_msg)   # Save current costmap to buffer


    def publish_obstacles(self, costmap_msg):	#TODO

        # Compute opticalFlowLK here.
        # previous image: self.prev_costmap_msg
        # current image: costmap_msg


        # generate obstacle_msg for TebLocalPlanner here.
        obstacle_msg = ObstacleArrayMsg() 
        obstacle_msg.header.stamp = rospy.Time.now()
        obstacle_msg.header.frame_id = self.global_frame[1:] \
            if self.global_frame[0]=='/' else self.global_frame

        # Add point obstacle
        obstacle_msg.obstacles.append(ObstacleMsg())
        obstacle_msg.obstacles[0].id = 0
        obstacle_msg.obstacles[0].polygon.points = [Point32()]
        obstacle_msg.obstacles[0].polygon.points[0].x = 1.5
        obstacle_msg.obstacles[0].polygon.points[0].y = 0
        obstacle_msg.obstacles[0].polygon.points[0].z = 0
	
        # Add line obstacle
        obstacle_msg.obstacles.append(ObstacleMsg())
        obstacle_msg.obstacles[1].id = 1
        line_start = Point32()
        line_start.x = -2.5
        line_start.y = 0.5
        line_end = Point32()
        line_end.x = -2.5
        line_end.y = 2
        obstacle_msg.obstacles[1].polygon.points = [line_start, line_end]
	
        # Add polygon obstacle
        obstacle_msg.obstacles.append(ObstacleMsg())
        obstacle_msg.obstacles[1].id = 2
        v1 = Point32()
        v1.x = -1
        v1.y = -1
        v2 = Point32()
        v2.x = -0.5
        v2.y = -1.5
        v3 = Point32()
	v3.x = 0
        v3.y = -1
        obstacle_msg.obstacles[2].polygon.points = [v1, v2, v3]

        self.obstacle_pub.publish(obstacle_msg)     # Publish predicted obstacles



if __name__ == '__main__':

    try:
        node = ObstaclePredictor()
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("node terminated.")

