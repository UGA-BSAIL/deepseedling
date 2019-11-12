'''
Author: Yu Jiang
Last update: 20190110

This program is to track seedlings in videos through detection results. 
Tracking results will be used for counting seedlings in videos. 
Compared with count_seedling_batch.py, this program will also save the
counted videos. 

Parameters to be changed:
video_folder: raw videos for counting
saving_folder: folder for saving counted videos
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
        save the counted videos in the saving folder

'''


import json
import numpy as np
import cv2
import time
import sys
import os
import os.path as osp

# change this according to the location of object detection module on your computer
sys.path.append('/home/yujiang/tfmodels/research/object_detection')

from utils import visualization_utils as vis_util
from utils import label_map_util
import sort
from PIL import Image


def main():
  # detection result folder
  detection_folder = '/media/yujiang/Data/Seedling/Experiments/counting_results_rgb/Final_model'
  # video folder
  video_folder = '/media/yujiang/Data/Seedling/Datasets/TestVideos'
  # folder for saving counted videos
  saving_folder = '/media/yujiang/Data/Seedling/Experiments/counting_videos_rgb/Final_model'
  # label map
  PATH_TO_LABELS = '/media/yujiang/Data/Seedling/Datasets/TFrecords/pascal_label_map.pbtxt'
  NUM_CLASSES = 2 # DO NOT change this, the number of classess identified by the detection model
  # Create label map
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  video_prefix_config_dict = {
    'TAMU_2015': {
      'iou_th':0.1, 'tracking_lifetime_th':9, 'is_transpose': False,
      'resize_fx':0.5, 'resize_fy':0.5, 'output_fps':5.0, 'is_display':True},
    'UGA_2015': {
      'iou_th':0.1, 'tracking_lifetime_th':15, 'is_transpose': False,
      'resize_fx':0.5, 'resize_fy':0.5, 'output_fps':5.0, 'is_display':True},
    'UGA_2018': {
      'iou_th':0.1, 'tracking_lifetime_th':9, 'is_transpose': True,
      'resize_fx':0.5, 'resize_fy':0.5, 'output_fps':5.0, 'is_display':True}}

  # Output video codec
  fourcc_codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

  for video_prefix, counting_config in video_prefix_config_dict.items():
    
    # tracking and counting parameters
    is_transpose = counting_config['is_transpose']
    iou_th = counting_config['iou_th']
    tracking_lifetime_th = counting_config['tracking_lifetime_th']
    resize_fx = counting_config['resize_fx']
    resize_fy = counting_config['resize_fy']
    output_fps = counting_config['output_fps']
    is_display = counting_config['is_display']
    # get testing videos with the video prefix
    video_list = os.listdir(video_folder)
    video_list = [f for f in video_list if f.startswith(video_prefix) and f.endswith('.mp4')]

    for video_file in video_list:
      video_name = video_file.split('.')[0]
      detection_file_path = osp.join(detection_folder, '{0}_detection.json'.format(video_name))
      video_file_path = osp.join(video_folder, video_file)
      output_video_filepath = osp.join(saving_folder, '{0}.avi'.format(video_file[:-4]))

      # tracking list
      seedling_id_list = list()
      seedling_lifetime = dict()
      # load detection results
      with open(detection_file_path,'r') as fid:
        jstr = json.load(fid)
        res_list = json.loads(jstr)

        # load video for rendering tracking result
        cap = cv2.VideoCapture(video_file_path)
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # generate output video writer
        fr_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fr_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if is_transpose:
          vd_writer = cv2.VideoWriter(output_video_filepath,
            fourcc_codec, output_fps,
            (int(fr_height*resize_fy), int(fr_width*resize_fx)))
        else:
          vd_writer = cv2.VideoWriter(output_video_filepath,
            fourcc_codec, output_fps,
            (int(fr_width*resize_fx), int(fr_height*resize_fy)))

        # initialize the Kalman filter based tracking algorithm
        mot_tracker = sort.Sort(iou_th=iou_th)
        # process individual frames
        for i in range(frame_num-1):
          ret, frame = cap.read()
          frame = cv2.resize(frame,None,fx=resize_fx, fy=resize_fy, interpolation = cv2.INTER_CUBIC)
          image_np = frame
          if is_transpose:
            image_np = image_np.transpose((1,0,2))
          # load detection result for the current frame
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
          # update trackers
          trackers = mot_tracker.update(dets)
          # update tracker names
          frame_bbox_array_new = trackers[:,0:4]
          frame_cls_new = list()
          category_index_new = dict()
          for d_index, d in enumerate(trackers[:, 4]):
            if d not in seedling_id_list:
              seedling_id_list.append(d)
              # max_cls_id = len(seedling_id_list)
            cur_d = seedling_id_list.index(d)+1

            if cur_d not in list(seedling_lifetime.keys()):
              seedling_lifetime[cur_d] = list()
            seedling_lifetime[cur_d].append(i)
            frame_cls_new.append(cur_d) 
            category_index_new[cur_d] = {'id':d_index, 'name':'Plant'+str(cur_d)}

          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              frame_bbox_array_new,
              frame_cls_new,
              frame_scores,
              category_index_new, # previously use 'category_index'
              use_normalized_coordinates=True,
              line_thickness=2)
          # write to the output video
          vd_writer.write(image_np)
          # 
          if is_display is True:
            cv2.imshow('video_window',image_np)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        vd_writer.release()
        cv2.destroyAllWindows()
        print('Processed the video {0} and saved as {1}.\n'.format(
          video_file_path, output_video_filepath))


if __name__ == "__main__":
  main()