# 1. Instruction 

- # Usecase 1: Analysis of single polyp image

1. After the project is successfully imported and the environment is established (see [instructions](https://github.com/Myosotis1111/Polyp-diagnosis-platform/tree/main)) **and run main.py**.

2. upload images by **clicking "Upload Images"** button with corresponding window location text files. Sample images can be found in [demo_io/demoImg](https://github.com/Myosotis1111/Polyp-diagnosis-platform/tree/main/Diagnosis_Platform/demo_io/demoImg). After successfully uploaded, the polyp image will be shown in the current tab widget.

3. **Clicking "Segment" button** to start the segmentation process. After the segmentation process completed, the current tab widget shows the masked image. 

4. **Clicking "Classify" button** to start the feature extraction and classification process. After the classification process completed, the current tab widget shows the image with enhanced edges.

5. If users have ground truth mask in the [demoGT](https://github.com/Myosotis1111/Polyp-diagnosis-platform/tree/main/Diagnosis_Platform/demo_io/demoGT) and the images are named including the true label (the number after second "-" indicates the JNET type), **"Evaluate" button can be clicked** to show the segmentation and classification results. The current tab widget shows the overlay mask image.

6. By **Clicking "Export Result" button**, users can choose to export classification results and segmented mask.

- # Usecase 2: Batch processing of multiple polyp images

1. When running the platform, **Click "Batch Processing" button**, the platform will start processing polyp images in [batch_io/batch_input](https://github.com/Myosotis1111/Polyp-diagnosis-platform/tree/main/Diagnosis_Platform/batch_io/batch_input). The segmented masks will be saved in the results folder. Also, an excel file including feature values and classification outcome for each polyp image will also be exported.

# 2. Description of files

- # Folders

1. [batch_io](https://github.com/Myosotis1111/Polyp-diagnosis-platform/tree/main/Diagnosis_Platform/batch_io): stores the inputs and outputs for batch processing (Usecase 2).

2. [classifier training](https://github.com/Myosotis1111/Polyp-diagnosis-platform/tree/main/Diagnosis_Platform/classifier%20training): contains the python script to train the classifier.

3. [demo_io](https://github.com/Myosotis1111/Polyp-diagnosis-platform/tree/main/Diagnosis_Platform/demo_io): stores the inputs and outputs for demonstration of the platform function.

4. [feature_data_for_cls](https://github.com/Myosotis1111/Polyp-diagnosis-platform/tree/main/Diagnosis_Platform/feature_data_for_cls): stores the training data and test data in excel formats for the training of the classificaiton model.

5. [utils](https://github.com/Myosotis1111/Polyp-diagnosis-platform/tree/main/Diagnosis_Platform/utils)/[lib](https://github.com/Myosotis1111/Polyp-diagnosis-platform/tree/main/Diagnosis_Platform/lib)/[pretrained_pth](https://github.com/Myosotis1111/Polyp-diagnosis-platform/tree/main/Diagnosis_Platform/pretrained_pth): these folders contain Polyp-PVT related codes. Detailed information can be refer to [link](https://github.com/DengPingFan/Polyp-PVT).

6. [model_weight_path](https://github.com/Myosotis1111/Polyp-diagnosis-platform/tree/main/Diagnosis_Platform/model_weight_path): contains weights of the segmentation and classification model. There are several classification models for choice.

    **The default classification model is SVM+SFBS.**

- # Python files

1. [main.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/detection_platform/main.py): the main program, entrance of the detection platform.

1. [evaluation.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/detection_platform/evaluation.py): entrance of the evaluation module. This program is designed for evaluating the performance of model on a certain dataset.

1. [detection_model_load](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/detection_platform/detection_model_load.py).py: for loading the yolov5 algorithm for the detection platform.

1. [model.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/detection_platform/model.py): the logic module of the MVC model, in charge of all calculation, detection and logic determination. The methods can be reused in evaluation.py.

1. [view.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/detection_platform/view.py): the GUI and controller module of the MVC model, in charge of action listening and GUI updating.

1. [signal.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/detection_platform/signal.py): the signals emitted by model.py to tell view module to update GUI are defined here.

1. [window.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/detection_platform/window.py): the Window class is defined here, with attributes and getter, setter methods.

1. [setup.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/detection_platform/setup.py): transform the python project into an executable software (if needed).

- # Other files

1. [Detection Platform.ui](detection_platform/Detection Platform.ui): the GUI of this detection platform, can be accessed via qt Designer.

1. [environment.yaml](detection_platform/environment.yaml): used for set up the environment on a new computer, for detailed information, please refer to the [readme file](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision#python-project) in the homepage.

1. [logo.ico](detection_platform/logo.ico): logo of THL, used as the icon for this project.

