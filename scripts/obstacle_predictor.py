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
        self.prediction_horizon = rospy.get_param("/obstacle_predictor/prediction_horizon")
        self.tol = rospy.get_param("/obstacle_predictor/movement_tolerence")

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
        I1g = np.reshape(
        self.prev_costmap_msg.data,
            [self.prev_costmap_msg.info.height, self.prev_costmap_msg.info.width]
        )
        I2g = np.reshape(costmap_msg.data, [costmap_msg.info.height, costmap_msg.info.width])
        # 7.31 add
        window_size = 3
        opt_uv = opticalFlowLK(I1g, I2g, window_size) # opt_uv = (u, v)
	
	    # Compute obstacle velocity
	    dt = costmap_msg.header.stamp - self.prev_costmap_msg.header.stamp
        robot_vel = (
            (costmap_msg.info.origin.position.x - self.prev_costmap_msg.info.origin.position.x)/dt,
            (costmap_msg.info.origin.position.y - self.prev_costmap_msg.info.origin.position.y)/dt
        )
        obstacle_vels = (
            opt_uv[0]/dt - robot_vel[0],
            opt_uv[1]/dt - robot_vel[1]
        )


        # Generate obstacle_msg for TebLocalPlanner here.
        obstacle_msg = ObstacleArrayMsg()
        obstacle_msg.header.stamp = rospy.Time.now()
        obstacle_msg.header.frame_id = self.global_frame[1:] \
            if self.global_frame[0]=='/' else self.global_frame

        for i in range(obstacle_vels[0].shape[0]):
            for j in range(obstacle_vels[0].shape[1]):
                # Add point obstacle to the message if obstacle speed is larger than tolerance.
                obstacle_speed = np.linalg.norm([obstacle_vels[0][i, j] + obstacle_vels[1][i, j]])
                if obstacle_speed > self.tol:
                    num_points = int(round(obstacle_speed * self.prediction_horizon / costmap_msg.info.resolution))
                    flow_vector_position = (
                        costmap_msg.info.origin.position.x + costmap_msg.info.resolution*(i+window_size/2),
                        costmap_msg.info.origin.position.y + costmap_msg.info.resolution*(j+window_size/2)
    	            )
                    normalized_vel = (
                        costmap_msg.info.resolution * obstacle_vels[0][i, j] / obstacle_speed,
                        costmap_msg.info.resolution * obstacle_vels[1][i, j] / obstacle_speed
                    )
                    for k in range(num_position):
                        obstacle_msg.obstacles.append(ObstacleMsg())
                        obstacle_msg.obstacles[-1].id = len(obstacle_msg.obstacles)-1
                        obstacle_msg.obstacles[-1].polygon.points = [Point32()]
                        obstacle_msg.obstacles[-1].polygon.points[0].x = flow_vector_position[0] + normalized_vel[0]*(k+1)
                        obstacle_msg.obstacles[-1].polygon.points[0].y = flow_vector_position[1] + normalized_vel[1]*(k+1)
                        obstacle_msg.obstacles[-1].polygon.points[0].z = 0

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

