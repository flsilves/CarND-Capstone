from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2 
import rospy
from scipy.stats import mode

GRAPH_PATH = 'light_classification/ssd_mobilenet_v2.pb'
SCORE_THRESHOLD = 0.80

class_lookup = {
        1 : TrafficLight.GREEN,
        2 : TrafficLight.YELLOW,
        3 : TrafficLight.RED,
}
class TLClassifier(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(GRAPH_PATH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_np = np.expand_dims(image, axis=0)

        with tf.Session(graph=self.graph) as sess:                
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                                feed_dict={self.image_tensor: image_np})
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        detection_idxs = np.argwhere(scores > SCORE_THRESHOLD)

        if len(detection_idxs) < 1:
            return TrafficLight.UNKNOWN
        else:
            class_modal = int(mode(classes[detection_idxs])[0][0][0])
            return class_lookup[class_modal]
        