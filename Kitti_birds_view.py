import numpy as np
import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from cv_bridge import CvBridge
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import random
from random import *
import ros_numpy
from PIL import Image, ImageEnhance
import pcl
import pcl_helper
class Birdeye():
    def __init__(self):
        self._pub = rospy.Publisher('result', PointCloud2, queue_size=1)
        self.count = 0
        self.rgb_map = 0.1
    def normalize_depth(self, val, min_v, max_v):
        """
        print 'normalized depth value'
        normalize values to 0-255 & close distance value has high value. (similar to stereo vision's disparity map)
        """
        return (((max_v - val) / (max_v - min_v)) * 255).astype(np.uint8)

    def in_range_points(self,points, x, y, z, x_range, y_range, z_range):
        """ extract in-range points """
        return points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], \
                                             y < y_range[1], z > z_range[0], z < z_range[1]))]




    def velo_points_2_top_view(self, points, x_range, y_range, z_range, scale):

        # Projecting to 2D
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        dist = np.sqrt(x ** 2 + y ** 2)

        # extract in-range points
        x_lim = self.in_range_points(x, x, y, z, x_range, y_range, z_range)
        y_lim = self.in_range_points(y, x, y, z, x_range, y_range, z_range)
        dist_lim = self.in_range_points(dist, x, y, z, x_range, y_range, z_range)

        # * x,y,z range are based on lidar coordinates
        x_size = int((y_range[1] - y_range[0]))
        y_size = int((x_range[1] - x_range[0]))

        # convert 3D lidar coordinates(vehicle coordinates) to 2D image coordinates
        # Velodyne coordinates info : http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        # scale - for high resolution
        x_img = -(y_lim * scale).astype(np.int32)
        y_img = -(x_lim * scale).astype(np.int32)

        # shift negative points to positive points (shift minimum value to 0)
        x_img += int(np.trunc(y_range[1] * scale))
        y_img += int(np.trunc(x_range[1] * scale))

        # normalize distance value & convert to depth map
        max_dist = np.sqrt((max(x_range)**2) + (max(y_range)**2))
        dist_lim = self.normalize_depth(dist_lim, min_v=0, max_v=max_dist)

        # array to img
        img = np.zeros([y_size * scale + 1, x_size * scale + 1], dtype=np.uint8)
        img[y_img, x_img] = dist_lim

        return img
    def callback(self, msg):
        scan = ros_numpy.numpify(msg)
        print(np.shape(scan))
        velo_points = np.zeros((scan.shape[-1],4))
        velo_points[:,0] = scan['x']
        velo_points[:,1] = scan['y']
        velo_points[:,2] = scan['z']
        #velo_points[:,:,3] = scan['intensity']


        # Plot result
        top_image = self.velo_points_2_top_view(velo_points, x_range=(-10, 10), y_range=(-10, 10), z_range=(-10, 10), scale=30)

        #top_image = Image.fromarray(top_image)
        #save_dir = '//media/imad/Windows/practice/rosbag-file/KITTI_RAW_ROSBAG/KITTI-Raw_bag_files/training_dataset'
        top_image = cv2.applyColorMap(top_image, cv2.COLORMAP_JET)
        #top_image=cv2.resize(top_image,(256,256))
        print(np.shape(top_image))
        cv2.imwrite('/media/imad/Windows/practice/paper/new/transfer_learning/data_maipulation/Output_images/{}.png'.format(self.count), top_image)
        #top_image.save(save_dir + "top_image%d.jpg" % self.count)
        self.count = self.count + 1
        #cv2.imshow('os',top_image)
        #cv2.waitKey(1)
    def main(self):
        rospy.Subscriber('/kitti/velo/pointcloud', PointCloud2, self.callback, queue_size=1)
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('bird_conv')
    tensor = Birdeye()
    tensor.main()
