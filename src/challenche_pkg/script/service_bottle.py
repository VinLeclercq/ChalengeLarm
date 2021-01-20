#! /usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse # you import the service message python classes generated from Empty.srv.
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

rospy.init_node('service_server') 
my_service = rospy.Service('/bottle', Empty , my_callback)
pose_result = Pose()
pose_sub = rospy.Subscriber('/move_base/feedback', MoveBaseActionFeedback, move_base_pose_callback)
rospy.spin()