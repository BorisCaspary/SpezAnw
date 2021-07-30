# Documentation

### Contents
* [Description](#Description)
* [Usage](#Usage)
* [Theory](#Theory)
  * [Nodes](#Nodes)
  * [Networks](#Networks)
* [Implementation](#Implementation)
  * [ROS](#ROS)
    * [Camera](#Camera)
    * [Processor](#Processor)
    * [Controller](#Controller)
    * [AI_Server](#AI_Server)
  * [Network](#Network)
* [Experiments](#Experiments)
  * [SGD](#SGD)
  * [Adam](#Adam)
  * [Adadelta](#Adadelta)
  * [Result](#Result)


## Description
This program uses a neural net to predict the values in images of handwritten, single digit numbers. In the current implementation, images from the MNIST-dataset are used and the prediction is compared to the true value.

## Usage
Use docker to run the following image for optimal results:
```sh
docker run -it --env="DISPLAY=10.181.36.38:0.0" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --env="XAUTHORITY=/home/boris/.Xauthority" --volume="$XAUTH:/home/boris/.Xauthority" -v "C:\Users\boris\Documents\Studium\8. Semester\KI":"/home/KI" --rm deepprojects/ros-ai:ss21 bash
```
Create a catkin workspace and place the program files into its *src* directory.
Source the ROS setup and run catkin:
```sh
source /opt/ros/noetic/setup.bash
catkin_make
```
Then source the setup in the devel folder:
```sh
source devel/setup.bash
```
Now you can start the program via roslaunch:
```sh
roslaunch talker_listener talker_listener.launch
```
On your screen you will see the controller node comparing the predicted values of the ai with the correct ones.

## Theory
### Nodes
ROS is based on the publisher-subscriber principle. Both of these roles are filled by nodes. A node is a single purpose program that can publish its output to a topic or subscribe to a topic for input.
The communication between nodes and topics uses messages. A message is a collections of predefined fields and potentially headers. Messages define what information can be communicated between nodes and topics.
Topics only allow communication to go in one direction: From publisher to topic to subscribers.
Another way to communicate in ROS are services. Services are similar to topics, but they use special service messages, which have a predefined request and response. Thus services allow communication in both directions.

### Networks
The training process of a neural net consists of two steps: Forward propagation and backpropagation.
During the forward propagation, an element of the training dataset is fed to the input of the net. It is then processed and an output given.
The so called loss is calculated from the difference between the output of the net and the ground truth.
Then the backpropagation happens, during which the network adjusts its weights according to the loss.
Both of these steps are repeated in alternation, until the network can predict the result accurately.

## Implementation

### ROS
![RQT_GRAPH](rqt_graph.png)
For some reason, services are not displayed in the graph.
The program is based on the talker_listener example from the beginner tutorial and simply expands on its python scripts.

#### Camera
The camera node takes a random MNIST-image and publishes it to the topic *image_pub*.
The image message is imported from sensor_msgs.msg. Using cv2, an image is read and converted into an image message via cv_bridge. This image is then published.
Simultaniously, the value of the number in the image is published to the topic *int_pub*.
The integer message uses the custom made IntWithHeader, to include a header.
These actions then repeat.
Issues between docker and windows concerning time synchronization can affect the synchronization used in the controller node. To combat this, the repetition rate can be set to 1Hz. 
[Source describing the problem](www.thorsten-hans.com/docker-on-windows-fix-time-synchronization)
(The fix described in the source does not seem to work)


#### Processor
The processor node subscribes to the *image_pub* topic and processes the image to ensure that the ai-predictions works.
It then converts it into a cv2 image via cv_bridge.
The cv2 image is converted to greyscale and resized to fit the format of the MNIST-dataset using cv2 and then converted back into an image message.
This message is then published to *proc_pub*.

#### Controller
The controller node subscribes to both the *proc_pub* and the *int_pub* topics. It then synchronizes the image with the number.
The image is sent as a request to the service *ai_service*. The response contains the prediction of the ai.
The response is compared to the number from the *int_pub* topic and the results are printed to the screen.

#### AI_Server
The ai_server node is connected to the *ai_service* service. When it receives an images, it uses the neural network to predict the value of the number in the image.
The result of the prediction is then sent back as a response of the *ai_service*.

### Network
In this program, a neural net is trained do determine the numeric value of a handwritten digit using the MNIST-dataset.
The images of the dataset are grayscale and have the shape of 28x28 pixels. Thus the input layer has the same amount of pixels.
There are 10 different digits in the arabic number system (0-9). Thus the output layer has 10 neurons.
For the activation function, ReLU is used.
To determine the output, the logarithm of the softmax is used.
In the training pipeline, the model is trained for ten cycles using 200 random images of the dataset. The learning rate is 0.01.
The loss function used was negative likelihood loss.
Experiments using optimization algorithms were performed.
The optimization functions used were: stochastic gradient descent and adam optimization.

## Experiments
For the experiments, optimization algorithms were varied and the accuracy of the resulting net compared.

### SGD
For the first model, stochastic gradient descent was used.
The resulting accuracy was 94% after ten epochs.

### Adam
For the second model, adam optimization was used.
The resulting accuracy was 96%. This amount was reached after the first epoch.

### Adadelta
For the third model, adadelta optimization was used.
The resulting accuracy was 92%.

### Result
The second model turned out to be the quickest to train and the most accurate.
Therefore adam optimization is used for this program and the amount of epochs is reduced to one.
