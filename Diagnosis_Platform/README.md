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

1. [main.py](https://github.com/Myosotis1111/Polyp-diagnosis-platform/blob/main/Diagnosis_Platform/main.py): the main program, entrance of the detection platform.

2. [batch_evaluation.py](https://github.com/Myosotis1111/Polyp-diagnosis-platform/blob/main/Diagnosis_Platform/batch_evaluation.py): Based on the outputs of the batch processing (the excel report and segmented masks), the classifier and segmentation model can be evaluated by running this script.

3. [loadClassifier.py](https://github.com/Myosotis1111/Polyp-diagnosis-platform/blob/main/Diagnosis_Platform/loadClassifier.py)/[loadSegModel.py](https://github.com/Myosotis1111/Polyp-diagnosis-platform/blob/main/Diagnosis_Platform/loadSegModel.py): for loading classifier (including scaler and selector) and segmentation model.

4. [model.py](https://github.com/Myosotis1111/Polyp-diagnosis-platform/blob/main/Diagnosis_Platform/model.py): the logic module of the MVC model, in charge of all calculation, detection and logic determination. The methods can be reused in evaluation.py.

5. [view.py](https://github.com/Myosotis1111/Polyp-diagnosis-platform/blob/main/Diagnosis_Platform/view.py): the GUI and controller module of the MVC model, in charge of action listening and GUI updating.

6. [signal.py](https://github.com/Myosotis1111/Polyp-diagnosis-platform/blob/main/Diagnosis_Platform/signal.py): the signals emitted by model.py to tell view module to update GUI are defined here.

7. [polyp.py](https://github.com/Myosotis1111/Polyp-diagnosis-platform/blob/main/Diagnosis_Platform/polyp.py): the Polyp class is defined here, with attributes and getter, setter methods.

8. [feature_extraction](https://github.com/Myosotis1111/Polyp-diagnosis-platform/blob/main/Diagnosis_Platform/feature_extraction.py): the logic for extracting features from the polyp images after segmentation, and setting the attribute values of current polyp instance.

- # Other files

1. [demo.ui](https://github.com/Myosotis1111/Polyp-diagnosis-platform/blob/main/Diagnosis_Platform/demo.ui): the GUI of this polyp diagnosis platform, can be accessed via qt Designer.

2. [environment.yaml](detection_platform/environment.yaml): used for set up the environment on a new computer, for detailed information, please refer to the [readme file](https://github.com/Myosotis1111/Polyp-diagnosis-platform/tree/main) in the homepage.

