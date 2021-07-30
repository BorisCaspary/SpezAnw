#!/usr/bin/env python
##based on listener from talker_listener example
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

bridge = CvBridge()

def callback(data):
    gray_pub = rospy.Publisher("proc_pub", Image, queue_size=10)
    
    rospy.loginfo(rospy.get_caller_id() + ' Received image')
    
    img = bridge.imgmsg_to_cv2(data)
    # tranformation: fit image into the format of mnist images
    # here not really necessary, because only mnist images are used
    # but just to be sure it's here
    proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = cv2.resize(proc, (28, 28))
    msg = bridge.cv2_to_imgmsg(img)

    rospy.loginfo(rospy.get_caller_id() + ' Sending processed image')
    gray_pub.publish(msg)

def processor():
    rospy.init_node('processor', anonymous=True)
    rospy.loginfo("Starting to listen")

    rospy.Subscriber('image_pub', Image, callback)

    rospy.spin()

if __name__ == '__main__':
    processor()
