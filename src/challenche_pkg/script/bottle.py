#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Empty, EmptyResponse # you import the service message python classes generated from Empty.srv.
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import LaserScan
from move_base_msgs.msg import MoveBaseAction, MoveBaseActionFeedback
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from math import * 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry


class LoadFeature(object):
    
    #initialisation
    def __init__(self):
    
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.camera_callback)
        self.bridge_object = CvBridge()
        self.x = 4
        self.list_pos = []
        self.pose_result = Pose()
        self.bottle = Point()
        self.bottle_pub = rospy.Publisher('/bottle', Point, queue_size=1)
        
    #recuperation de la position du robot
    def my_position_callback(self, msg):
        self.pose_result = msg.pose.pose

    #retourne les coordonees de la bouteilles visualisee avec en entree l image que percoit le robot
    def get_position(self, img):
        #initialisation de valeurs
        distance = 0.0
        bottle_pos = Point()
        is_in_list = False
        approx = 2.0

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v= cv2.split(hsv)
        ret_h, th_h = cv2.threshold(h,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Remplissage des contours (equivalent a un operateur morpho de Fermeture)
        im_floodfill = th_h.copy()
        h, w = th_h.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        th = th_h | im_floodfill_inv

        # Detection des objets
        _,contours,_ = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for i in range (0, len(contours)) :
            mask_BB_i = np.zeros((len(th),len(th[0])), np.uint8)
            x,y,w,h = cv2.boundingRect(contours[i])
            cv2.drawContours(mask_BB_i, contours, i, (255,255,255), -1)
            BB_i=cv2.bitwise_and(img,img,mask=mask_BB_i)
        
        #print(w, h)
        
        # Recuperation de la disatance de l objet par rapport au robot
        if ((w>10) and (h>10)) :
            if (w>h) : distance = float(54)/w #54 est la hauteur d une cannette debout a 1x du robot
            else : distance =float(54)/h
            #print(distance)

            # Recuperation de l'angle de la vision du robot par rapport a l axe de la map
            quaternion_x = float(self.pose_result.orientation.x)
            quaternion_y = self.pose_result.orientation.y
            quaternion_z = self.pose_result.orientation.z
            quaternion_w = self.pose_result.orientation.w
            _,_,angle = euler_from_quaternion([quaternion_x, quaternion_y, quaternion_z, quaternion_w])

            #calcul de la position de la bouteille
            bottle_pos.x = self.pose_result.position.x + float(distance) * cos(angle)
            bottle_pos.y = self.pose_result.position.y + float(distance) * sin(angle)
            bottle_pos.z = 0

            #
            if self.list_pos :
                for pos in self.list_pos :
                    if ((bottle_pos.x - approx <= pos.x <= bottle_pos.x + approx) and (bottle_pos.y - approx <= pos.y <= bottle_pos.y + approx)) :
                        #print(" I know this can")
                        is_in_list = True
                    #else : print("I don't know this can") 

        if is_in_list == False and distance <= 1.5 :
            print("new bottle") 
            self.list_pos.append(bottle_pos)
            print bottle_pos
            return bottle_pos
        else : return Point()


    def camera_callback(self,data):
        #recuperation de l image de la camera du robot
        try:
            #We select bgr8 because its the OpenCV encoding by default
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
            print("pas d image en entree de camera")
        
        #recuperation de l image de la cannette pour la comparaison
        try:
            image_1 = cv2.imread('/home/user/catkin_ws/src/challenche_pkg/pictures/bottle.jpg',1)
        except CvBridgeError as e:
            print(e)
            print("pas d'image dans le dossier, placez vous devant une cannette et tapez 'rosrun challenche_pkg pictures.py'")
        
        image_1 = cv2.resize(image_1,(300,100))
        image_2 = cv2.resize(cv_image,(300,200))

        gray_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)

        #creation du masque rouge
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

        except:
            pass  
        else: 
            #retourne les coordonnees du robot
            pose_sub = rospy.Subscriber('/odom', Odometry, self.my_position_callback)
            #recuperation des coordonnees de la bouteille
            self.bottle = self.get_position(image_2)
            if (self.bottle != Point()) :
                self.bottle_pub.publish(self.bottle)
        cv2.waitKey(1)

    def prove(self):
            for self.x in range(4,1001,18):
                for y in range (1,500):
                    rospy.sleep(0.0001)


def main():
    rospy.init_node('load_feature_node', anonymous=True)
    load_feature_object = LoadFeature()
    load_feature_object.prove()
    try:
        rospy.spin()
        
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()