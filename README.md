Mask Detection Using CNN . . 

Method :
https://medium.com/@samosirmasniari/68f5d9347e23

-Prepare the Dataset:
 https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

extract the dataset folder, put it into your project.

-Install

run train_masker.py make sure to load the dataset in your directory.

After running, the output folder ‘temp_dataset’ and model_detector_cnn.h5. make sure it is still in 1 directory of your project.

MaskDetectionUsing-CNN/

├── dataset/                # Folder for dataset (with_mask, without_mask)

├── models/                 # Saved trained models

├── scripts/                # Python scripts for training and inference

│   ├── train.py            # Script to train the CNN model

│   ├── detect.py           # Script for real-time mask detection


-Running Interface. 
cnn model processing last step run the realtime_masker.py programme. make sure model_detector_cnn.h5 is in your project directory.






