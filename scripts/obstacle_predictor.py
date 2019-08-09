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
        self.movement_tol = rospy.get_param("/obstacle_predictor/movement_tolerence")
        self.timediff_tol = rospy.get_param("/obstacle_predictor/timediff_tolerence")
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


    def spin(self):
        rospy.spin()


    def globalCostmapCallback(self, msg):
        '''
        Save corrent global costmap to buffer
        '''
        self.global_costmap_msg = reshapeCostmap(msg)


    def localCostmapCallback(self, msg):
        '''
        Compute optical flow of local costmap to predict velocities of moving obstacles.
        Predicted velocities will be used for generating ObstacleArrayMsg for TebLocalPlannerROS.
        '''
        if isOccupancyGrid(self.global_costmap_msg):
            msg = reshapeCostmap(msg)
            self.mask_costmap(msg)

            if isOccupancyGrid(self.prev_local_costmap_msg):
                # Compute opticalFlowLK here.
                dt = msg.header.stamp.to_sec() - self.prev_local_costmap_msg.header.stamp.to_sec()
                if dt < self.timediff_tol: # skip opticalflow when dt is larger than self.timediff_tol (sec).
                    I1g, I2g = self.preprocess_images(msg)
                    flow, rep_physics = opticalFlowLK(self.prev_local_costmap_msg.data, msg.data, self.window_size)
            
                    # Generate and Publish ObstacleArrayMsg
                    self.publish_obstacles(msg, dt, flow, rep_physics)

            self.prev_local_costmap_msg = copy(msg)   # Save current costmap to buffer


    def publish_obstacles(self, costmap_msg, dt, flow, rep_physics):
        '''
        Generate and publish ObstacleArrayMsg from flow vectors.
        '''
        obstacle_vels = np.transpose(flow, axes=[1,0,2]) / dt * costmap_msg.info.resolution # opt_uv needs to be transposed. ( [y,x] -> [x,y] )
        obstacle_vels[:, :, 0][costmap_msg.data.T==0] = 0   # Mask obstacle velocity using costmap occupancy.
        obstacle_vels[:, :, 1][costmap_msg.data.T==0] = 0

        # Generate obstacle_msg for TebLocalPlanner here.
        obstacle_msg = ObstacleArrayMsg()
        obstacle_msg.header.stamp = rospy.Time.now()
        obstacle_msg.header.frame_id = self.global_frame[1:] \
            if self.global_frame[0]=='/' else self.global_frame

        for i in range(obstacle_vels.shape[0]):
            for j in range(obstacle_vels.shape[1]):
                # Add point obstacle to the message if obstacle speed is larger than self.movement_tol.
                obstacle_speed = np.linalg.norm([obstacle_vels[i, j, 0] + obstacle_vels[i, j, 1]])
                if obstacle_speed > self.movement_tol:
                    flow_vector_position = (
                        costmap_msg.info.origin.position.x + costmap_msg.info.resolution*(i+0.5),
                        costmap_msg.info.origin.position.y + costmap_msg.info.resolution*(j+0.5)
                    )
                    obstacle_msg.obstacles.append(ObstacleMsg())
                    obstacle_msg.obstacles[-1].id = len(obstacle_msg.obstacles)-1
                    obstacle_msg.obstacles[-1].polygon.points = [Point32(), Point32()]
                    obstacle_msg.obstacles[-1].polygon.points[0].x = flow_vector_position[0]
                    obstacle_msg.obstacles[-1].polygon.points[0].y = flow_vector_position[1]
                    obstacle_msg.obstacles[-1].polygon.points[0].z = 0
                    obstacle_msg.obstacles[-1].polygon.points[1].x = flow_vector_position[0] + obstacle_vels[i, j, 0]*self.prediction_horizon
                    obstacle_msg.obstacles[-1].polygon.points[1].y = flow_vector_position[1] + obstacle_vels[i, j, 1]*self.prediction_horizon
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


    def preprocess_images(self, costmap_msg):
        '''
        Preprocess images for optical flow.
        Match the location of current costmap and previous costmap.
        '''
        w = costmap_msg.info.width
        h = costmap_msg.info.height
        dx = costmap_msg.info.origin.position.x - self.prev_local_costmap_msg.info.origin.position.x
        dy = costmap_msg.info.origin.position.y - self.prev_local_costmap_msg.info.origin.position.y

        di = int(round(dy/costmap_msg.info.resolution))
        dj = int(round(dx/costmap_msg.info.resolution))

        img_prev = np.zeros_like(self.prev_local_costmap_msg.data)
        img_curr = np.zeros_like(costmap_msg.data)
        if di < 0:
            if dj < 0:
                img_prev[-di:h, -dj:w] = self.prev_local_costmap_msg.data[:h+di, :w+dj]
                img_curr[-di:h, -dj:w] = costmap_msg.data[-di:h, -dj:w]
            else:
                img_prev[-di:h, :w-dj] = self.prev_local_costmap_msg.data[:h+di, dj:w]
                img_curr[-di:h, :w-dj] = costmap_msg.data[-di:h, :w-dj]
        else:
            if dj < 0:
                img_prev[:h-di, -dj:w] = self.prev_local_costmap_msg.data[di:h, :w+dj]
                img_curr[:h-di, -dj:w] = costmap_msg.data[:h-di, -dj:w]
            else:
                img_prev[:h-di, :w-dj] = self.prev_local_costmap_msg.data[di:h, dj:w]
                img_curr[:h-di, :w-dj] = costmap_msg.data[:h-di, :w-dj]
        
        return img_prev, img_curr

##  End of class ObstaclePredictor.



def opticalFlowLK(I1g, I2g, window_size, tau=1e-2): # 7.31 add
    '''
    Lucas-Kanade optical flow
    '''
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
    uv = np.zeros([I1g.shape[0], I1g.shape[1], 2])
    # within window window_size * window_size
    rep_x_list = []
    rep_y_list = []
    u_list = []
    v_list = []
    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            b = -It # Av = b <- Ix*u+Iy*v+It = 0
            A = np.stack([Ix, Iy], axis = 1) # 8.1 add. Ix & Iy are columns.
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            nu = np.matmul(np.linalg.pinv(A), b) # v = (A'A)^-1 * (-A'It)
            if 0.3<np.linalg.norm(nu)<1.1:
                uv[i, j, 0] = nu[0]
                uv[i, j, 1] =-nu[1]
                u_list.append(nu[0])
                v_list.append(nu[1])
                rep_x_list.append(j)
                rep_y_list.append(i)
                #print(i,j)

    rep_x = np.average(rep_x_list)
    rep_y = np.average(rep_y_list)
    rep_u = np.average(u_list)
    rep_v = np.average(v_list)
 
    return uv, [rep_x,rep_y,rep_u,-rep_v]   # [August 8, 19:46] Return changed: (ndarray(nx,ny), ndarray(nx,ny)) -> ndarray(nx,ny,2) 


def isOccupancyGrid(msg):
    '''
    Return False if msg is not an OccupancyGrid class.
    '''
    return type(msg) == type(OccupancyGrid())


def reshapeCostmap(msg):
    '''
    Reshape, remove negative values and change type as uint8 (for cv2.resize)
    '''
    temp = copy(msg)
    temp.data = np.reshape(
        msg.data,
        [msg.info.height, msg.info.width]
    ).clip(0).astype(np.uint8)
    return temp


if __name__ == '__main__':

    try:
        node = ObstaclePredictor()
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("node terminated.")

