#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Empty, EmptyResponse # you import the service message python classes generated from Empty.srv.
import service_bottle.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import Pose
from move_base_msgs.msg import MoveBaseAction, MoveBaseActionFeedback

def my_callback(request):
    global pose_result
    print "[ BOTTLE ] Found at :"
    print pose_result
    return EmptyResponse()

def move_base_pose_callback(msg):
    global pose_result
    pose_result = msg.feedback.base_position.pose


class LoadFeature(object):

    def __init__(self):
    
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.camera_callback)
        self.bridge_object = CvBridge()
        self.x = 4

    def camera_callback(self,data):
        try:
            # We select bgr8 because its the OpenCV encoding by default
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        
        try:
            image_1 = cv2.imread('/home/user/catkin_ws/src/challenche_pkg/pictures/bottle.jpg',1)
        except CvBridgeError as e:
            print(e)
        
        image_1 = cv2.resize(image_1,(300,100))
        image_2 = cv2.resize(cv_image,(300,200))

        gray_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)

        max_red = np.array([255,200,200])
        min_red = np.array([40,0,0])

        hsv = cv2.cvtColor(image_2, cv2.COLOR_BGR2HSV)
        mask_r = cv2.inRange(hsv, min_red, max_red)
        res_r = cv2.bitwise_and(image_2, image_2, mask= mask_r)

        
        #Initialize the ORB Feature detector 
        orb = cv2.ORB_create(nfeatures = 1000)

        #Make a copy of th eoriginal image to display the keypoints found by ORB
        #This is just a representative
        preview_1 = np.copy(image_1)
        preview_2 = np.copy(res_r)

        #Create another copy to display points only
        dots = np.copy(image_1)

        #Extract the keypoints from both images
        train_keypoints, train_descriptor = orb.detectAndCompute(gray_1, None)
        test_keypoints, test_descriptor = orb.detectAndCompute(res_r, None)

        #Draw the found Keypoints of the main image
        cv2.drawKeypoints(image_1, train_keypoints, preview_1, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.drawKeypoints(image_1, train_keypoints, dots, flags=2)

        try:
            #############################################
            ################## MATCHER ##################
            #############################################

            #Initialize the BruteForce Matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

            #Match the feature points from both images
            matches = bf.match(train_descriptor, test_descriptor)

            #The matches with shorter distance are the ones we want.
            matches = sorted(matches, key = lambda x : x.distance)
            #Catch some of the matching points to draw
            
                
            good_matches = matches[:] # THIS VALUE IS CHANGED YOU WILL SEE LATER WHY 
            

            #Parse the feature points
            train_points = np.float32([train_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            test_points = np.float32([test_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

            #Create a mask to catch the matching points 
            #With the homography we are trying to find perspectives between two planes
            #Using the Non-deterministic RANSAC method
            M, mask = cv2.findHomography(train_points, test_points, cv2.RANSAC,5.0)

            #Catch the width and height from the main image
            h,w = gray_1.shape[:2]

            #Create a floating matrix for the new perspective
            pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

            #Create the perspective in the result 
            dst = cv2.perspectiveTransform(pts,M)

            cv2.imshow('Points',preview_1)
            
            cv2.imshow('Detection',res_r)   
            cv2.imshow('image',image_2)
        except :
            pass  
        else : 
            #bottle_found = rospy.ServiceProxy('/bottle', Empty)
            #bottle_found()
            pose_sub = rospy.Subscriber('/move_base/feedback', MoveBaseActionFeedback, move_base_pose_callback)
        cv2.waitKey(1)

    def prove(self):
            for self.x in range(4,1001,18):
                for y in range (1,500):

                   # print (self.x)
                    rospy.sleep(0.0001)


#def main():

my_service = rospy.Service('/bottle', Empty , my_callback)
pose_result = Pose()
load_feature_object = LoadFeature()
rospy.init_node('detection_bottle_server', anonymous=True)
load_feature_object.prove()
try:
    rospy.spin()
    
except KeyboardInterrupt:
    print("Shutting down")
cv2.destroyAllWindows()

#if __name__ == '__main__':