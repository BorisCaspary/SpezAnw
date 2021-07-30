#!/usr/bin/env python

import rospy
import rospkg
from cv_bridge import CvBridge
import torch
from torchvision import transforms

import Net
from talker_listener.srv import AI, AIResponse

n = Net.__dict__["context"]["Net"]

bridge = CvBridge()
rospack = rospkg.RosPack()

device = "cpu"

model = n(input_shape=[1, 28, 28]).to(device)
transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])

def idx_of_max(values):
    result = 0
    for idx, i in enumerate(values):
        if i > values[result]:
            result = idx
    return result

def handle_ai(req):
    orig = bridge.imgmsg_to_cv2(req.image)
    img = transform(orig.copy())
    
    result = model(img).int()
    prediction = idx_of_max(result[0].tolist())
    rospy.loginfo("Returning integer " + str(result))
    return AIResponse(int(prediction))

def ai_server():
    rospy.init_node('ai_server')
    s = rospy.Service('ai_service', AI, handle_ai)
    rospy.loginfo("Ready to receive")
    rospy.spin()

if __name__ == "__main__":
    state_dict = torch.load(rospack.get_path('talker_listener') + '/scripts/trained_models/mnist_simple_fc.pth')
    model.load_state_dict(state_dict)
    ai_server()
