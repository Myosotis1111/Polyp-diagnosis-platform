# 1. Instruction 

- # Usecase 1: Analysis of single polyp image

1. After the project is successfully imported and the environment is established (see [instructions](https://github.com/Myosotis1111/Polyp-diagnosis-platform/tree/main)) **and run main.py**.

2. upload images by **clicking "Upload Images"** button with corresponding window location text files. Sample images can be found in [demo_io](https://github.com/Myosotis1111/Polyp-diagnosis-platform/tree/main/Diagnosis_Platform/demo_io/demoImg). After successfully uploaded, the polyp image will be shown in the current tab widget.

3. **Clicking "Segment" button** to start the segmentation process. After the segmentation process completed, the current tab widget shows the masked image. 

4. **Clicking "Classify" button** to start the feature extraction and classification process. After the classification process completed, the current tab widget shows the image with enhanced edges.

5. If users have ground truth mask in the [demoGT](https://github.com/Myosotis1111/Polyp-diagnosis-platform/tree/main/Diagnosis_Platform/demo_io/demoGT) and the images are named including the true label (the number after second "-" indicates the JNET type), **"Evaluate" button can be clicked** to show the segmentation and classification results. The current tab widget shows the overlay mask image.

6. By **Clicking "Export Result" button**, users can choose to export classification results and segmented mask.

- # For Developer: evaluation

1. **Select [evaluation.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/detection_platform/evaluation.py)**, configure the folder_path to the path of tested data (default: testset) . parameters like confidence threshold and NMS IOU threshold can be also set here. **Run [evaluation.py](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/blob/main/detection_platform/evaluation.py)** after setting.

2. The test report will then be provided under [evaluation](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/tree/main/detection_platform/evaluation) folder in the form of excel file.

- # For Developer: release executable version

1. If an executable version is needed for users to use the platform without python. First make sure that cx_Freeze is installed, if not, open the command line to execute:

    `pip install cx_Freeze`

2. After successfully installed cx_Freeze, change the directory to the python project and execute:

    `python setup.py build`

    Then a folder named "build" containing executable file is established.
# 2. Description of files

- # Folders

1. [evaluation](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/tree/main/detection_platform/evaluation): stores the images and result excel file obtained during the evaluation process (via running evaluation.py)

1. [images](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/tree/main/detection_platform/images): stores the images obtained during the detection process (via running main.py).

1. [testset](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/tree/main/detection_platform/testset): contains all original images in the dataset. Users can try these images for testing the platform.

1. [utils](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/tree/main/detection_platform/utils)/[model](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/tree/main/detection_platform/model): these folders contain yolov5 related codes.

1. [weights](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/tree/main/detection_platform/weights): contains weights of the detection model. There are several models for choice:


    - classic yolo: default yolov5 model trained with CIoU and data augmentation.
    - simAM: yolov5 model with Focal-EIoU, SimAM attention module and data augmentation.
    - simAM_lite: yolov5 model with Focal-EIoU, SimAM attention module, but without data augmentation.

    **the default model is simAM. If users want to change the model, copy the best.pt in the desired folder and replace the best.pt in the root path of weights folder.**

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

