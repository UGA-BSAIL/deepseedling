# Author: Yu Jiang
# Last update: 2018-12-12
# Purpose:
# This program is to generate the bash script for exporting all models to frozen graph
#

import os
import os.path as osp
import pandas as pd
import shutil

# change these parameters accoding to your own project
exp_config_filepath = '/media/yujiang/Data/Seedling/Experiments/Summary_result.xlsx'
base_folder = '/media/yujiang/Data/Seedling/Experiments/model'


def export_header_script():
    script_str = '#!/bin/bash\n' + \
        '# export all trained models to frozen graphs\n' + \
        'cd ~/tfmodels/research\n' + \
        'echo "Start to export the models"\n'
    return script_str

def model_export_script(input_type, config_path, ckpt_path, export_dir):
    script_str = 'INPUT_TYPE="{0}"\nPIPELINE_CONFIG_PATH="{1}"\nTRAINED_CKPT_PREFIX="{2}"\nEXPORT_DIR="{3}"\n'.format(
        input_type, config_path, ckpt_path, export_dir) + \
        'python object_detection/export_inference_graph.py --input_type=${INPUT_TYPE}' + \
        ' --pipeline_config_path=${PIPELINE_CONFIG_PATH}' + \
        ' --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX}' + \
        ' --output_directory=${EXPORT_DIR}\n\n'
    return script_str


def main():
    # change this accoding to your own project
    script_filepath = '/media/yujiang/Data/Seedling/Experiments/export_all_models.sh'

    with open(script_filepath, 'w') as f:
        f.write(export_header_script())
        input_type = 'image_tensor'
        exp_configs = pd.read_excel(exp_config_filepath)
        for exp_id, exp_config in exp_configs.iterrows():
            print('Exp name: {0}, Best Check Point Step: {1}.\n'.format(exp_config.Experiment, exp_config.BestCheckPoint))
            exp_folder_path = osp.join(base_folder, exp_config.Experiment)

            best_ckpt_prefix = 'model.ckpt-{0}'.format(exp_config.BestCheckPoint)
            best_ckpt_prefix_path = osp.join(exp_folder_path, best_ckpt_prefix)

            config_path = osp.join(base_folder, '{0}.config'.format(exp_config.Experiment))

            export_dir = osp.join(base_folder, exp_config.Experiment, 'frozen_model')
            if not osp.isdir(export_dir):
                os.mkdir(export_dir)
            else:
                shutil.rmtree(export_dir)
                os.mkdir(export_dir)
                

            export_script_str = model_export_script(input_type, config_path, best_ckpt_prefix_path, export_dir)

            f.write('echo "Start to export model for {0}"\n'.format(exp_config.Experiment))
            f.write(export_script_str)





if __name__ == '__main__':
    main()
