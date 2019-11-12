# Author: Yu Jiang
# Last update: 2018-12-12
# Purpose:
# This program is to generate the bash script for model inference
#

import os
import os.path as osp
import pandas as pd

# change these parameters accoding to your own project
exp_config_filepath = '/media/yujiang/Data/Seedling/Experiments/Summary_result.xlsx'
base_folder = '/media/yujiang/Data/Seedling/Experiments/model'
result_folder = '/media/yujiang/Data/Seedling/Experiments/inference_results'
data_folder = '/media/yujiang/Data/Seedling/Datasets/TFrecords'

def inference_header_script():
    script_str = '#! /bin/bash\n' + \
        '# evaluate the exported frozen graphs\n' + \
        'cd ~/tfmodels/research\n\n'
    return script_str

def inference_script(input_record_path, output_record_path, frozen_graph_path):
    script_str = 'python object_detection/inference/infer_detections.py' + \
        ' --input_tfrecord_paths={0}'.format(input_record_path) + \
        ' --output_tfrecord_path={0}'.format(output_record_path) + \
        ' --inference_graph={0}'.format(frozen_graph_path) + \
        ' --discard_image_pixels\n\n'
    return script_str


def main():

    # change this accoding to your own project
    inference_script_filepath = '/media/yujiang/Data/Seedling/Experiments/inference.sh'
    with open(inference_script_filepath, 'w') as f:
        f.write(inference_header_script())

        exp_configs = pd.read_excel(exp_config_filepath)
        for exp_id, exp_config in exp_configs.iterrows():
            print('Exp name: {0}, Best Check Point Step: {1}.\n'.format(exp_config.Experiment, exp_config.BestCheckPoint))
            exp_folder_path = osp.join(base_folder, exp_config.Experiment)
            
            f.write('# Evaluation on validation datasets for {0}\n'.format(exp_config.Experiment))
            f.write('echo "Evaluation on validation datasets for {0}"\n'.format(exp_config.Experiment))
            frozen_graph_path = osp.join(exp_folder_path, 'frozen_model', 'frozen_inference_graph.pb')
            # for validation datasets
            val_dataset = exp_config.eval_input_reader
            val_dataset_names = val_dataset.split(' ')
            for val_dataset_name in val_dataset_names:
                input_record_path = osp.join(data_folder, '{0}.tfrecord'.format(val_dataset_name))
                output_record_path = osp.join(
                    result_folder, '{0}_val_{1}.tfrecord'.format(exp_config.Experiment, val_dataset_name))
                f.write('# Validation dataset {0}\n'.format(val_dataset_name))
                f.write('echo "Validation dataset {0}"\n'.format(val_dataset_name))
                f.write(inference_script(input_record_path, output_record_path, frozen_graph_path))
            # for test datasets
            test_dateset = exp_config.Test
            test_dateset_names = test_dateset.split(' ')
            f.write('# Evaluation on test datasets for {0}\n'.format(exp_config.Experiment))
            f.write('echo "Evaluation on test datasets for {0}"\n'.format(exp_config.Experiment))
            for test_dataset_name in test_dateset_names:
                input_record_path = osp.join(data_folder, '{0}.tfrecord'.format(test_dataset_name))
                output_record_path = osp.join(
                    result_folder, '{0}_test_{1}.tfrecord'.format(exp_config.Experiment, test_dataset_name))
                f.write('# Test dataset {0}\n'.format(test_dataset_name))
                f.write('echo "Test dataset {0}"\n'.format(test_dataset_name))
                f.write(inference_script(input_record_path, output_record_path, frozen_graph_path))



if __name__ == '__main__':
    main()