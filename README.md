# Enhancing Medical Diagnosis using Deep-learning based Segmentation

# Author：Yang Xinchen

# Master thesis:

**(Updated on 08.09.2024)**

Final version of the complete thesis uploaded.
               
![Software GUI](software_screenshot.png)

# Python project：

- # The folder "[detection_platform](https://git.mylab.th-luebeck.de/xinchen.yang/building-management-machine-vision/-/tree/main/detection_platform)" contains all the source codes used in this bachelor thesis.

The python project （see folder “**detection_platform**”） is packed using conda. Users can download the folder (please make sure there is **at least 8GB** available in the disk Anaconda installed), import to IDE and follow three steps to configure the environment:

**(1) Use Conda to create a new environment**. Run the following command in a terminal or command prompt:

    conda env create -f environment.yaml

This will create a new Conda environment using the configuration in the environment.yaml file.

**(2)** After the environment is created successfully, **activate the newly created environment.** Run the following command:

    conda activate rs_detection

**(3)** Finally, **change the Python interpreter** of the program to the interpreter in the conda environment you just created. The project can then be run via **main.py**.

# Additional modules

**Folder "preprocessing" and "Extra Functions" are not related to the grading of the bachelor thesis.** They serve as supportive codes for data preprocessing and software optimization.

# Dataset

The dataset used for training and evaluation can be access via the link below:

https://cloud.th-luebeck.de/index.php/apps/files/?dir=/Bachelor_ECUST/20_Dataset&fileid=4892230

# Executable Software (no python environment needed):

For common users, an executable software version is provided with all the features. It can be downloaded via the link below:

https://cloud.th-luebeck.de/index.php/apps/files/?dir=/Bachelor_ECUST/30_GUI&fileid=4120537

**(1)** To install it, unzip the zip file to any path except desktop (unknown error may occur). 

**(2)** To run it, run the .exe file in the folder "rs detection platform ver1.2".

**Note:** Folder "testset" contains images as samples to upload and detect.
