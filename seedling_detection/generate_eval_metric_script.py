# Author: Yu Jiang
# Last update: 2018-12-12
# Purpose:
# This program is to generate the bash script for calculate the performance metrics for the detection results
#

import os
import os.path as osp
import pandas as pd

# change these parameters accoding to your own project
exp_config_filepath = '/media/yujiang/Data/Seedling/Experiments/Summary_result.xlsx'
base_folder = '/media/yujiang/Data/Seedling/Experiments/model'
result_folder = '/media/yujiang/Data/Seedling/Experiments/inference_results'
config_folder = '/media/yujiang/Data/Seedling/Experiments/inference_metric_configs'
label_map_path = '/media/yujiang/Data/Seedling/Datasets/TFrecords/pascal_label_map.pbtxt'

METRIC_DICT = {'PASCAL': 'pascal_voc_detection_metrics', 'COCO':'coco_detection_metrics'}

def header_script():
    script_str = '#! /bin/bash\n' + \
        '# evaluate the detection results\n' + \
        'cd ~/tfmodels/research\n\n'
    return script_str

def calculate_metric_script(eval_dir, eval_config_path, input_config_path):
    script_str = 'python object_detection/metrics/offline_eval_map_corloc.py' + \
        ' --eval_dir={0}'.format(eval_dir) + \
        ' --eval_config_path={0}'.format(eval_config_path) + \
        ' --input_config_path={0}\n\n'.format(input_config_path)
    return script_str

def export_eval_config(config_path, metric_str):
    with open(config_path, 'w') as cf:
        cf.write('metrics_set: "{0}"'.format(metric_str))

def export_input_config(config_path, input_data_path, label_dict_path):
    with open(config_path, 'w') as cf:
        cf.write('label_map_path: "{0}"\n'.format(label_dict_path))
        cf.write('tf_record_input_reader: {{input_path: "{0}" }}\n'.format(
            input_data_path))

def main():
    eval_detection_script_filepath = '/media/yujiang/Data/Seedling/Experiments/eval_detections.sh'
    with open(eval_detection_script_filepath, 'w') as f:
        f.write(header_script())

        exp_configs = pd.read_excel(exp_config_filepath)
        for exp_id, exp_config in exp_configs.iterrows():
            print('Exp name: {0}, Best Check Point Step: {1}.\n'.format(exp_config.Experiment, exp_config.BestCheckPoint))
            exp_folder_path = osp.join(base_folder, exp_config.Experiment)
            
            f.write('# Calculate performance metrics for {0}\n'.format(exp_config.Experiment))
            f.write('echo "Calculate performance metrics for {0}"\n'.format(exp_config.Experiment))
            frozen_graph_path = osp.join(exp_folder_path, 'frozen_model', 'frozen_inference_graph.pb')
            # for validation datasets
            val_dataset = exp_config.eval_input_reader
            val_dataset_names = val_dataset.split(' ')
            for val_dataset_name in val_dataset_names:
                # detection result tfrecord
                detection_record_path = osp.join(
                    result_folder, '{0}_val_{1}.tfrecord'.format(exp_config.Experiment, val_dataset_name))
                for metric_key, metric_val in METRIC_DICT.items():
                    # input config file
                    input_config_path = osp.join(config_folder, '{0}_val_{1}_{2}_input_config.pbtxt'.format(
                        exp_config.Experiment, val_dataset_name, metric_key))
                    export_input_config(input_config_path, detection_record_path, label_map_path)
                    # eval config file
                    eval_config_path = osp.join(config_folder, '{0}_val_{1}_{2}_eval_config.pbtxt'.format(
                        exp_config.Experiment, val_dataset_name, metric_key))
                    export_eval_config(eval_config_path, metric_val)
                    # script for calculation of performance metric
                    eval_metric_filepath = osp.join(result_folder, '{0}_val_{1}_{2}.csv'.format(
                        exp_config.Experiment, val_dataset_name, metric_key))
                    f.write('# Using {0} detection metrics\n'.format(metric_key))
                    f.write('echo "Using {0} detection metrics"\n'.format(metric_key))
                    f.write(calculate_metric_script(eval_metric_filepath, eval_config_path, input_config_path))

            # for test datasets
            test_dateset = exp_config.Test
            test_dateset_names = test_dateset.split(' ')
            f.write('echo "Calculate performance metrics for {0}"\n'.format(exp_config.Experiment))
            for test_dataset_name in test_dateset_names:
                # detection result tfrecord
                detection_record_path = osp.join(
                    result_folder, '{0}_test_{1}.tfrecord'.format(exp_config.Experiment, test_dataset_name))
                for metric_key, metric_val in METRIC_DICT.items():
                    # input config file
                    input_config_path = osp.join(config_folder, '{0}_test_{1}_{2}_input_config.pbtxt'.format(
                        exp_config.Experiment, test_dataset_name, metric_key))
                    export_input_config(input_config_path, detection_record_path, label_map_path)
                    # eval config file
                    eval_config_path = osp.join(config_folder, '{0}_test_{1}_{2}_eval_config.pbtxt'.format(
                        exp_config.Experiment, test_dataset_name, metric_key))
                    export_eval_config(eval_config_path, metric_val)
                    # script for calculation of performance metric
                    eval_metric_filepath = osp.join(result_folder, '{0}_test_{1}_{2}.csv'.format(
                        exp_config.Experiment, test_dataset_name, metric_key))
                    f.write('# Using {0} detection metrics\n'.format(metric_key))
                    f.write('echo "Using {0} detection metrics"\n'.format(metric_key))
                    f.write(calculate_metric_script(eval_metric_filepath, eval_config_path, input_config_path))



if __name__ == '__main__':
    main()