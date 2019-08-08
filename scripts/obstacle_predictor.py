#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import copy
from scipy import signal, interpolate

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
        self.global_costmap_topic = rospy.get_param("/obstacle_predictor/global_costmap_topic")
        self.local_costmap_topic = rospy.get_param("/obstacle_predictor/local_costmap_topic")
        self.obstacle_topic = rospy.get_param("/obstacle_predictor/obstacle_topic")
        self.prediction_horizon = rospy.get_param("/obstacle_predictor/prediction_horizon")
        self.tol = rospy.get_param("/obstacle_predictor/movement_tolerence")
        self.window_size = rospy.get_param("/obstacle_predictor/window_size")

        # Initialize ros node
        rospy.init_node('obstacle_predictor', anonymous=True)

        # Costmap buffer
        self.prev_local_costmap_msg = None
        self.global_costmap_msg = None

        # Publisher and subscriber
        self.global_costmap_sub = rospy.Subscriber(self.global_costmap_topic, OccupancyGrid, self.globalCostmapCallback)
        self.local_costmap_sub = rospy.Subscriber(self.local_costmap_topic, OccupancyGrid, self.localCostmapCallback)
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


    def globalCostmapCallback(self, msg):

        self.global_costmap_msg = copy(msg)   # Save current costmap to buffer
        self.global_costmap_msg.data = np.reshape(
            self.global_costmap_msg.data,
            [msg.info.height, msg.info.width]
        )


    def localCostmapCallback(self, msg):

        self.resize_and_mask_costmap(msg)

        self.publish_obstacles(msg)
        self.prev_local_costmap_msg = copy(msg)   # Save current costmap to buffer


    def publish_obstacles(self, costmap_msg):

        if isOccupancyGrid(self.global_costmap_msg) and isOccupancyGrid(self.prev_local_costmap_msg): # Check costmap buffers.

            # Compute opticalFlowLK here.
            flow = opticalFlowLK(self.prev_local_costmap_msg, costmap_msg, self.window_size)
	
            # Compute obstacle velocity
            dt = costmap_msg.header.stamp.to_sec() - self.prev_costmap_msg.header.stamp.to_sec()
            if dt < 1.0: # skip opticalflow when dt is larger than 1 sec.
                robot_vel = (
                    (costmap_msg.info.origin.position.x - self.prev_costmap_msg.info.origin.position.x)/dt,
                    (costmap_msg.info.origin.position.y - self.prev_costmap_msg.info.origin.position.y)/dt
                )
                obstacle_vels = np.transpose(flow, axes=[1,0,2]) / dt * costmap_msg.info.resolution # opt_uv needs to be transposed. ( [y,x] -> [x,y] )

                # Generate obstacle_msg for TebLocalPlanner here.
                obstacle_msg = ObstacleArrayMsg()
                obstacle_msg.header.stamp = rospy.Time.now()
                obstacle_msg.header.frame_id = self.global_frame[1:] \
                    if self.global_frame[0]=='/' else self.global_frame

                for i in range(obstacle_vels.shape[0]):
                    for j in range(obstacle_vels.shape[1]):
                        # Add point obstacle to the message if obstacle speed is larger than tolerance.
                        obstacle_speed = np.linalg.norm([obstacle_vels[i, j, 0] + obstacle_vels[i, j, 1]])
                        if obstacle_speed > self.tol:
                            num_points = int(round(obstacle_speed * self.prediction_horizon / costmap_msg.info.resolution))
                            flow_vector_position = (
                                costmap_msg.info.origin.position.x + costmap_msg.info.resolution*(i+self.window_size/2),
                                costmap_msg.info.origin.position.y + costmap_msg.info.resolution*(j+self.window_size/2)
                            )
                            normalized_vel = (
                                costmap_msg.info.resolution * (obstacle_vels[i, j, 0]-robot_vel[0]) / obstacle_speed,
                                costmap_msg.info.resolution * (obstacle_vels[i, j, 1]-robot_vel[1]) / obstacle_speed
                            )
                            if normalized_vel > self.tol:
                                for k in range(num_points): # num_position -> num_points
                                    obstacle_msg.obstacles.append(ObstacleMsg())
                                    obstacle_msg.obstacles[-1].id = len(obstacle_msg.obstacles)-1
                                    obstacle_msg.obstacles[-1].polygon.points = [Point32()]
                                    obstacle_msg.obstacles[-1].polygon.points[0].x = flow_vector_position[0] + normalized_vel[0]*(k+1)
                                    obstacle_msg.obstacles[-1].polygon.points[0].y = flow_vector_position[1] + normalized_vel[1]*(k+1)
                                    obstacle_msg.obstacles[-1].polygon.points[0].z = 0

                self.obstacle_pub.publish(obstacle_msg)     # Publish predicted obstacles

        ''' End of function ObstaclePredictor.publish_obstacles '''


    def resize_and_mask_costmap(self, costmap_msg):
        '''
        Resize costmap data to 2D array and mask static obstacles.
        '''

        costmap_msg.data = np.reshape(
            costmap_msg.data,
            [costmap_msg.info.height, costmap_msg.info.width]
        )

        dx = costmap_msg.info.origin.position.x - self.global_costmap.info.origin.position.x
        dy = costmap_msg.info.origin.position.y - self.global_costmap.info.origin.position.y

        multiplier = float(costmap_msg.info.resolution)/float(self.global_costmap_msg.info.resolution)

        di = int(round(dx/self.global_costmap_msg.info.resolution))
        dj = int(round(dy/self.global_costmap_msg.info.resolution))

        mask = cv2.resize(
            self.global_costmap_msg.data[
                di:di+int(round(costmap_msg.info.height*multiplier)),
                dj:dj+int(round(costmap_msg.info.width*multiplier))
            ],
            dsize=(costmap_msg.info.height, costmap_msg.info.width)
        )
        costmap_msg.data[mask!=0] = 0


    ''' End of class ObstaclePredictor '''



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
    uv = np.zeros(I1g.shape + [2])
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
            uv[i, j, 0]=nu[0]
            uv[i, j, 1]=nu[1]
 
    return uv    # [August 8, 19:46] Return changed: (ndarray(nx,ny), ndarray(nx,ny)) -> ndarray(nx,ny,2) 


def isOccupancyGrid(msg):
    ''' Return False if msg is not an OccupancyGrid class. '''

    return type(msg) == type(OccupancyGrid())


if __name__ == '__main__':

    try:
        node = ObstaclePredictor()
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("node terminated.")

