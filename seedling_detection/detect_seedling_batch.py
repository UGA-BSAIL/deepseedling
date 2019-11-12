'''
Author: Yu Jiang
Last update: 20190110

This program is to detect seedlings in videos. 
Detection results will be saved and used for seedling counting.

Parameters to be changed:
PATH_TO_CKPT: the path for a frozen Tensorflow detection model
PATH_TO_LABELS: the path of the category label map file
NUM_CLASSES: the number of categories defined in the dataset
data_folder: the path of the folder containing testing videos
output_folder: the path of the folder saving the detection results

video_prefix_config_dict: a dictionary object in which the key is the prefix of videos files
and the value is a dictionary object containing detection configuration. Currently, only one
parameter to be included, "is_transpose", indicating whether to transpose video frames.
An example is provided below for three video prefixes.
video_prefix_config_dict = {'TAMU_2015':{'is_transpose':False},
                            'UGA_2015':{'is_transpose':False},
                            'UGA_2018':{'is_transpose':True}}

Input: currently, the program only supports videos in MP4 format
Output: detections (bounding box, category, score) in a JSON file named with the corresponding video name


'''
import numpy as np
import os
import os.path as osp
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import json

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.10.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

if __name__ == "__main__":

  # Path to frozen detection graph. This is the actual model that is used for the object detection.
  PATH_TO_CKPT = '/media/yujiang/Data/Seedling/Experiments/model/Final_model/frozen_model/frozen_inference_graph.pb'

  # List of the strings that is used to add correct label for each box.
  PATH_TO_LABELS = '/media/yujiang/Data/Seedling/Datasets/TFrecords/pascal_label_map.pbtxt'

  # Number of classes in the dataset
  NUM_CLASSES = 2

  # Load pretrained model for object detection
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  # Create label map
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  # Size, in inches, of the output images.
  IMAGE_SIZE = (12, 8)

  
  data_folder = '/media/yujiang/Data/Seedling/Datasets/TestVideos'
  output_folder = '/media/yujiang/Data/Seedling/Experiments/counting_results_rgb/Final_model'
  all_files = os.listdir(data_folder)

  video_prefix_config_dict = {'TAMU_2015':{'is_transpose':False},
                              'UGA_2015':{'is_transpose':False},
                              'UGA_2018':{'is_transpose':True}}

  for video_prefix, config_dict in video_prefix_config_dict.items():

    all_video_files = [f for f in all_files if f.startswith(video_prefix) and f.endswith('.mp4')]

    print('Processing video list: {0}\n'.format(all_video_files))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    is_transpose = config_dict['is_transpose']

    for video_file in all_video_files:
      video_file_path = osp.join(data_folder, video_file)
      video_file_prefix = video_file.split('.')[0]
      video_detection_filename = video_file_prefix + '_detection.json'
      video_detection_file_path = osp.join(output_folder, video_detection_filename)
      print('Processing video {0}.\n'.format(video_file_path))

      # saving file path
      fid = open(video_detection_file_path,'w')
      # result list
      res_list = list()


      cap = cv2.VideoCapture(video_file_path)
      frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      print('There is {0} frames\n'.format(frame_num))

      # Initiate SIFT detector
      sift = cv2.xfeatures2d.SIFT_create()
      
      resize_fx = 0.5
      resize_fy = 0.5

      # while(cap.isOpened()):
      for i in range(frame_num-1):
        ret, frame = cap.read()
        frame = cv2.resize(frame,None,fx=resize_fx, fy=resize_fy, interpolation = cv2.INTER_CUBIC)
        image_np = frame
        resized_img_hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        v_image_eq = clahe.apply(resized_img_hsv[:, :, 2])
        resized_img_hsv[:, :, 2] = v_image_eq
        resized_img = cv2.cvtColor(resized_img_hsv, cv2.COLOR_HSV2BGR)
        resized_img_inference = cv2.cvtColor(resized_img_hsv, cv2.COLOR_HSV2RGB)
        if is_transpose:
          resized_img = resized_img.transpose((1,0,2))
          resized_img_inference = resized_img_inference.transpose((1,0,2))

        # Actual detection.
        output_dict = run_inference_for_single_image(resized_img_inference, detection_graph)
        # filter plant boxes
        plant_boxes = [tuple(output_dict['detection_boxes'][i].astype('float64')) 
                      for i in range(len(output_dict['detection_boxes'])) 
                      if output_dict['detection_classes'][i] == 1 and output_dict['detection_scores'][i] > 0.7]
        # detected_plant_cls = [None]*len(plant_boxes)
        plant_classes = [str(output_dict['detection_classes'][i])
                      for i in range(len(output_dict['detection_classes'])) 
                      if output_dict['detection_classes'][i] == 1 and output_dict['detection_scores'][i] > 0.7]
        plant_scores = [output_dict['detection_scores'][i].astype('float64') 
                      for i in range(len(output_dict['detection_scores'])) 
                      if output_dict['detection_classes'][i] == 1 and output_dict['detection_scores'][i] > 0.7]
        
        # save to a dict
        frame_dict = dict()
        frame_dict['frame'] = i
        frame_dict['bbox'] = plant_boxes
        frame_dict['box_cls'] = plant_classes
        frame_dict['box_scores'] = plant_scores
        res_list.append(frame_dict)

      en_str = json.dumps(res_list)
      json.dump(en_str, fid)

      cap.release()
      fid.flush()
      fid.close()