import rospy
from sensor_msgs.msg import LaserScan
from sklearn.cluster import DBSCAN
import numpy as np
import math

class DBSCANob:

    def __init__(self):
        rospy.init_node('hihi',anonymous=True)
        print('init')
        self.scanfiner_sub = rospy.Subscriber("/fined_scan",LaserScan,self.finedscanCallback)
    
    def spin(self):
        rospy.spin()
        
    def finedscanCallback(self, msg):
        # print(msg.ranges)
        # print(type(msg.ranges))
        self.temp = []
        for i in range(0, len(msg.ranges)):
            if msg.ranges[i] < 0.01:
                self.temp.append([0,0])
            else:
                self.temp.append([msg.ranges[i] * math.cos(- math.pi + 0.0174532923847 * i),msg.ranges[i] * math.sin(- math.pi + 0.0174532923847 * i)])
            
        print(self.temp)
                

        clustered = DBSCAN(eps = 0.2, min_samples = 7).fit(db.temp)
        #print(clustered.labels_)
        #print(self.temp)


db = DBSCANob()
db.spin()