#! /usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError

class Photo(object):

    def __init__(self):
    
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.img_callback)
        self.bridge_object = CvBridge()
        img_result = Image()
    
    def img_callback(self, msg):
        try:
            img_result = self.bridge_object.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        #une_Image = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_2/Course_images/test_image_1.jpg')
        cv2.imshow('image', img_result)
        cv2.imwrite('/home/user/catkin_ws/src/challenche_pkg/pictures/bottle.jpg',img_result)
        cv2.waitKey(250)
        rospy.spin()
        cv2.destroyAllWindows()

rospy.init_node('service_server')
showing_image_object = Photo()
