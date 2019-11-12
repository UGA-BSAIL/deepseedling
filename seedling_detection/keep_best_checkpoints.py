# Author: Yu Jiang
# Last update: 2018-12-12
# Purpose:
# This program is to keep the best checkpoint based on the validation accuracy
#

import os
import os.path as osp
import pandas as pd

# change these parameters accoding to your own project
exp_config_filepath = '/media/yujiang/Data/Seedling/Experiments/Summary_result.xlsx'
base_folder = '/media/yujiang/Data/Seedling/Experiments/model'

def main():
    exp_configs = pd.read_excel(exp_config_filepath)
    for exp_id, exp_config in exp_configs.iterrows():
        print('Exp name: {0}, Best Check Point Step: {1}.\n'.format(exp_config.Experiment, exp_config.BestCheckPoint))
        exp_folder_path = osp.join(base_folder, exp_config.Experiment)

        best_ckpt_dataname = 'model.ckpt-{0}.data-00000-of-00001'.format(exp_config.BestCheckPoint)
        best_ckpt_indexname = 'model.ckpt-{0}.index'.format(exp_config.BestCheckPoint)
        best_ckpt_metaname = 'model.ckpt-{0}.meta'.format(exp_config.BestCheckPoint)
        best_ckpt_filelist = [best_ckpt_dataname, best_ckpt_indexname, best_ckpt_metaname]

        model_filelist = os.listdir(exp_folder_path)
        model_filelist = [filename for filename in model_filelist if filename.startswith('model.ckpt')]

        for model_filename in model_filelist:
            if model_filename not in best_ckpt_filelist:
                model_filepath = osp.join(exp_folder_path, model_filename)
                os.remove(model_filepath)
                print('Removed {0}.\n'.format(model_filepath))




if __name__ == '__main__':
    main()