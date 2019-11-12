'''
Author: Yu Jiang
Last update: 20190110

This program is to track seedlings in videos through detection results. 
Tracking results will be used for counting seedlings in videos

Parameters to be changed:
PATH_TO_LABELS: the path of the category label map file
NUM_CLASSES: the number of categories defined in the dataset
detection_folder: the path of the folder containing detection results
video_prefix_config_dict: a dictionary object in which the key is a prefix of testing videos, 
and the value is a dictionary object containing counting configuration. Currently there are
two counting paramters: 
  iou_th: the minimum iou to accept the assignment of one detection to one existing tracker
  tracking_lifetime_th: the threshold of lifetime considering one tracker is valid in the video
An example of video_prefix_config_dict is provided below

video_prefix_config_dict = {'TAMU_2015': {'iou_th':0.1, 'tracking_lifetime_th':9},
                              'UGA_2015': {'iou_th':0.1, 'tracking_lifetime_th':15},
                              'UGA_2018': {'iou_th':0.1, 'tracking_lifetime_th':9}}

Input: detection results of videos
Output: counts of seedlings in individual videos (to the console)

'''

import json
import numpy as np
import cv2
import time
import sys
import os
import os.path as osp

# please also change this according to the location of your tensorflow models-object detection module
sys.path.append('/home/yujiang/tfmodels/research/object_detection')

from utils import visualization_utils as vis_util
from utils import label_map_util
import sort
from PIL import Image


def main():

  PATH_TO_LABELS = '/media/yujiang/Data/Seedling/Datasets/TFrecords/pascal_label_map.pbtxt'
  NUM_CLASSES = 2 # DO NOT change this

  detection_folder = '/media/yujiang/Data/Seedling/Experiments/counting_results_rgb/Final_model'
  video_folder = '/media/yujiang/Data/Seedling/Datasets/TestVideos'

  video_prefix_config_dict = {'TAMU_2015': {'iou_th':0.1, 'tracking_lifetime_th':9},
                              'UGA_2015': {'iou_th':0.1, 'tracking_lifetime_th':15},
                              'UGA_2018': {'iou_th':0.1, 'tracking_lifetime_th':9}}

  # video_prefix_config_dict = {'UGA_2015': {'iou_th':0.1, 'tracking_lifetime_th':15}}

  # Create label map
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  for video_prefix, counting_config in video_prefix_config_dict.items():
    # tracking and counting parameters
    iou_th = counting_config['iou_th']
    tracking_lifetime_th = counting_config['tracking_lifetime_th']
    # video list
    video_list = os.listdir(video_folder)
    video_list = [f for f in video_list if f.startswith(video_prefix) and f.endswith('.mp4')]
    for video_file in video_list:
      video_name = video_file.split('.')[0]
      detection_file_path = osp.join(detection_folder, '{0}_detection.json'.format(video_name))
      video_file_path = osp.join(video_folder, video_file)
      # tracking list
      seedling_id_list = list()
      seedling_lifetime = dict()
      # load detection results
      with open(detection_file_path,'r') as fid:
        jstr = json.load(fid)
        res_list = json.loads(jstr)
        cap = cv2.VideoCapture(video_file_path)
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # initialize the Kalman filter based tracking algorithm
        mot_tracker = sort.Sort(iou_th=iou_th)
        total_time = 0
        for i in range(frame_num-1):
          # detection result for the current frame
          detection_res = res_list[i]

          frame_bbox = detection_res['bbox']
          frame_bbox_array = np.array(frame_bbox, dtype=np.float32)
          frame_cls = detection_res['box_cls']
          frame_cls = np.array([int(i) for i in frame_cls])
          frame_scores = np.asarray(detection_res['box_scores']).reshape((-1,1))

          if len(frame_bbox_array) != 0:
            dets = np.concatenate((frame_bbox_array, frame_scores), axis=1)
          else:
            dets = np.array([])
          
          start_time = time.time()
          # update trackers based on the current detection result
          trackers = mot_tracker.update(dets)
          cycle_time = time.time() - start_time
          # update the total time used
          total_time += cycle_time
          
          # update the tracker list 
          for d_index, d in enumerate(trackers[:, 4]):
            if d not in seedling_id_list:
              seedling_id_list.append(d)
              # max_cls_id = len(seedling_id_list)
            cur_d = seedling_id_list.index(d)+1

            if cur_d not in list(seedling_lifetime.keys()):
              seedling_lifetime[cur_d] = list()
            seedling_lifetime[cur_d].append(i)
        # identify valid seedling trackers
        valid_seedling_tracker_list = list()
        for k, v in seedling_lifetime.items():
            if len(v) >= tracking_lifetime_th:
                valid_seedling_tracker_list.append(k)

      print('Video {0} has {1} seedlings using {2} seconds.\n'.format(
        video_file, len(valid_seedling_tracker_list), total_time))
   

  # cap = cv2.VideoCapture('/media/yujiang/HTP-2017/Cotton/2018/ENGR/20180613/ENGR1_1_19.MOV')


  
  # generate output video writer
  fr_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  fr_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  # if is_transpose:
  #   vd_writer = cv2.VideoWriter(output_video_filepath,
  #     fourcc_codec, 5.0,
  #     (int(fr_height*resize_fy), int(fr_width*resize_fx)))
  # else:
  #   vd_writer = cv2.VideoWriter(output_video_filepath,
  #     fourcc_codec, 5.0,
  #     (int(fr_width*resize_fx), int(fr_height*resize_fy)))

  

  

  
  # print(tracked_seedling_list)

  # print(mot_tracker.trackers)


if __name__ == "__main__":
  main()