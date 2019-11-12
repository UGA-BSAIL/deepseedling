# DeepSeedling
DeepSeedling: Deep convolutional network and Kalman filter for plant seedling detection and counting in the field

This repository contains the source code and instruction for running models trained for seedling detection. Detailed methodology and results can be found from our paper (Under review with Plant Methods).


1. Prerequisites

1.1. Deep learning framework configuration

Models used in the study were trained using Tensorflow 1.12 and Tensorflow models (https://github.com/tensorflow/models). Please install these deep learning framework before using the source code for seedling detection.


1.2. Pretrained Faster RCNN for seedling detection

Download the model pretrained for cotton seedling detection via the link https://figshare.com/s/b3bc56a13a5a801b0267.


1.3. Source code for seedling detection and counting

Download the source code from this repository to a local computer. 



2. Instructions for detection and tracking

Testing environment: Ubuntu 18.04, Python 3.6.7, NVIDIA GTX1080-Ti. 

2.1. Install necessary packages: tensorflow (tested in 1.12.0), tensorflow models, numba (0.40.0), opencv-python (3.4.2.17), opencv-contrib-python (3.4.2.17), pandas (0.23.4), scikit-image (0.14.0), and scikit-learn (0.20.0)

2.2. Copy the seedling_detection folder to the object_detection module of tensorflow models (e.g., /home/yujiang/tfmodels/research/object_detection).

2.3. Run detect_seedling_batch.py to detect seedlings in testing vidoes and save the detection results. Please see the program header for detailed explanation of changable parameters.

2.4. Run count_seedling_batch.py to track seedling detections and count the number of seedlings in testing videos. This program outputs the counting results in console without showing the tracking process. Please see the program header for detailed explanation of changable parameters.

2.5. Run visualize_seedling_tracking_batch.py to track seedling detections, count the number of seedlings, and save the counted results in testing videos. Please see the program header for detailed explanation of changable parameters.


3. Additional resources
Original seedling images/videos and annotations can be downloaded via the link https://figshare.com/s/616956f8633c17ceae9b. All annotations were made using MS VOTT 1.5 version.
