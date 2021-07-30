#!/usr/bin/env python
##based on talker from talker_listener example
import os
import rospy
import random
import cv2
import rospkg
from sensor_msgs.msg import Image
from talker_listener.msg import IntWithHeader
from cv_bridge import CvBridge

bridge = CvBridge()
rospack = rospkg.RosPack()
image_path = rospack.get_path('talker_listener') + '/img/'

def camera():
    image_pub = rospy.Publisher('image_pub', Image, queue_size=10)
    int_pub = rospy.Publisher('int_pub', IntWithHeader, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    # an issue between windows and docker can mess with the synchronization
    # in that case, use 1hz, otherwize 10hz
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        rand_int = random.randint(0, 9)
        image = cv2.imread(image_path + str(rand_int) + ".png")

        if not image is None:
            image_msg = bridge.cv2_to_imgmsg(image)

            int_msg = IntWithHeader()
            int_msg.data = rand_int
            
            rospy.loginfo("sending image")
            image_pub.publish(image_msg)

            rospy.loginfo("sending int: " + str(rand_int))
            int_pub.publish(int_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        camera()
    except rospy.ROSInterruptException:
        pass
