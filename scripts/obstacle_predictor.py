#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import copy
from scipy import signal # 7.31 add

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
        # 7.31 add
        # window_size = 3
        # opt_uv = opticalFlowLK(self.prev_costmap_msg, costmap_msg, window_size) # opt_uv = (u, v)



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


    def opticalFlowLK(I1g, I2g, window_size, tau=1e-2): # 7.31 add
 
        kernel_x = np.array([[-1., 1.], [-1., 1.]])
        kernel_y = np.array([[-1., -1.], [1., 1.]])
        kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
        w = window_size / 2 # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
        I1g = I1g / 255. # normalize pixels
        I2g = I2g / 255. # normalize pixels
        # Implement Lucas Kanade
        # for each point, calculate I_x, I_y, I_t
        mode = 'same'
        fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
        fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
        ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
        u = np.zeros(I1g.shape)
        v = np.zeros(I1g.shape)
        # within window window_size * window_size
        for i in range(w, I1g.shape[0]-w):
            for j in range(w, I1g.shape[1]-w):
                Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
                Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
                It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
                b = -It # Av = b <- Ix*u+Iy*v+It = 0
                A = np.stack([Ix, Iy], axis = 1) # 8.1 add. Ix & Iy are columns.
                # if threshold τ is larger than the smallest eigenvalue of A'A:
                nu = np.matmul(np.linalg.pinv(A), b) # v = (A'A)^-1 * (-A'It)
                u[i,j]=nu[0]
                v[i,j]=nu[1]
 
        return (u,v)


if __name__ == '__main__':

    try:
        node = ObstaclePredictor()
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("node terminated.")

