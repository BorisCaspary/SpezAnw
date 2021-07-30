#!/usr/bin/env python

import rospy
import cv2
import message_filters as mf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from talker_listener.msg import IntWithHeader
from talker_listener.srv import AI, AIResponse

bridge = CvBridge()

def callback(image, integ):
    correct_value = integ.data
    rospy.loginfo("Saved image")

    rospy.wait_for_service("ai_service")

    try:
        ai_service = rospy.ServiceProxy("ai_service", AI)
        result = ai_service(image)
        rospy.loginfo("Correct value " + str(correct_value))
        rospy.loginfo("AI returned   " + str(result.result))
        rospy.loginfo("The prediction was " + str(result.result == correct_value).upper())
        
    except rospy.ServiceException as e:
        rospy.loginfo("Service call failed: %s"%e)

def controller():
    rospy.init_node('controller', anonymous=True)
    rospy.loginfo("Starting to control")

    image_sub = mf.Subscriber('proc_pub', Image)
    int_sub = mf.Subscriber('int_pub', IntWithHeader)

    ts = mf.TimeSynchronizer([image_sub, int_sub], 10)
    ts.registerCallback(callback)

    rospy.spin()

if __name__ == '__main__':
    controller()
