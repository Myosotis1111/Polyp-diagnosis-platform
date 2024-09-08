import os
import re
import shutil
import warnings
import cv2
import mlxtend
import sklearn
import torch.nn.functional as F
import pydicom
import torch
from PIL import Image
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import QFileDialog, QMessageBox
from loadSegModel import load_seg_model
from loadClassifier import load_cls_model
from polyp import Polyp
from signal import MySignals
from feature_extraction import ImageFeatureExtractor
import numpy as np
import pandas as pd
import time


class Model:

    def __init__(self):

        warnings.filterwarnings("ignore", category=UserWarning, message=".*nn.functional.upsample.*")
        warnings.simplefilter(action='ignore', category=np.VisibleDeprecationWarning)
        warnings.simplefilter(action='ignore', category=np.RankWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)

        self.batch_input_dir = 'batch_io/batch_input'
        self.batch_output_dir = 'batch_io/batch_result'
        self.export_filename = 'test_features.xlsx'
        self.batch_gt_mask_dir = 'batch_io/batch_gt_mask'
        self.demo_output_dir = 'demo_io/demo_result'

        self.cls_model = "SVM+SFBS"  # can use 6 combinations
        self.classifier_model_path = f"model_weight_path/{self.cls_model}/weight.pkl"
        self.scaler_path = f"model_weight_path/{self.cls_model}/scaler.pkl"
        self.selector_path = f"model_weight_path/{self.cls_model}/selector.pkl"
        self.feature_mean_excel_path = "feature_data_for_cls/feature_means.xlsx"

        self.segModel_path = 'model_weight_path/seg_model_pth/PolypPVT_best_dice_0.9116.pth'
        self.demo_gt_path = 'demo_io/demoGT'

        self.ms = MySignals()
        self.polyp = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.segModel = load_seg_model(self.segModel_path, self.device)
        self.testsize = 640
        self.sorted_importance = None
        self.feature_names = []
        # self.mode = "feature only"  # activate to skip segmentation and classification, only extracting features
        # self.mode = "classifier only"  # activate to skip segmentation and use gt mask (if exists) for classification
        self.mode = "standard"  # default setting

        self.clsModel, self.scaler, self.selector = load_cls_model(self.classifier_model_path, self.scaler_path,
                                                                   self.selector_path)

        # Calculate feature contributions
        if hasattr(self.clsModel, 'feature_importances_'):
            # For tree-based models
            importances = self.clsModel.feature_importances_
        elif hasattr(self.clsModel, 'coef_'):
            # For linear models, such as linear SVM
            importances = np.abs(self.clsModel.coef_[0])
        else:
            importances = None

        if importances is not None:
            if self.selector is not None:
                if isinstance(self.selector, sklearn.base.TransformerMixin):
                    # If selector is from sklearn
                    selected_features = self.selector.get_feature_names_out()
                elif isinstance(self.selector, mlxtend.feature_selection.SequentialFeatureSelector):
                    # If selector is from mlxtend
                    selected_indices = self.selector.k_feature_idx_
                    # Create feature names as x1, x2, ..., xn
                    selected_features = [f'x{i + 1}' for i in selected_indices]
                else:
                    raise ValueError("Unsupported selector type")
            else:
                # If no selector, use all feature names assuming the features are in the same order as the importances
                selected_features = [f'x{i + 1}' for i in range(len(importances))]

            # Create a dictionary of feature importances
            feature_importances = dict(zip(selected_features, importances))

            # Sort features by importance
            self.sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        else:
            self.sorted_importances = []

        # Load the Excel file into a DataFrame
        df = pd.read_excel(self.feature_mean_excel_path, index_col=0)

        # Extract columns and rows
        columns = df.columns
        index = df.index

        # Define attributes based on column and row headers
        for row in index:
            for col in columns:
                # Construct the attribute name
                attr_name = f"{col}_{row}"
                # Get the cell value
                value = df.at[row, col]

                # Set the attribute dynamically
                setattr(self, attr_name, value)

                self.feature_names.append(attr_name)

    def upload(self):

        testsize = self.testsize

        file_path, _ = QFileDialog.getOpenFileName(None, 'Choose file', '', '*.jpg *.png *.tif *.jpeg *.dcm *.bmp')

        if file_path:
            # Get file extension
            _, ext = os.path.splitext(file_path)

            if ext.lower() == '.dcm':
                # Read DICOM file
                ds = pydicom.dcmread(file_path)
                image = ds.pixel_array
            else:
                # Read TIFF file or other image files
                image = np.array(Image.open(file_path))

            # Convert to RGB format (if grayscale)
            if len(image.shape) == 2:
                image = np.stack((image,) * 3, axis=-1)

            # Resize image to the testsize
            image_resized = Image.fromarray(image).resize((testsize, testsize), Image.LANCZOS)

            # Convert image to RGB format in memory
            png_image = image_resized.convert('RGB')

            # Create Polyp object
            self.polyp = Polyp()

            self.polyp.set_png_img(png_image)

            # Set the file name
            self.polyp.set_file_name(os.path.basename(file_path))

            print(f"Image uploaded and processed: {file_path}")

            # Convert PIL Image to NumPy array
            png_image_array = np.array(png_image)

            # Get image dimensions
            height, width, channels = png_image_array.shape
            bytesPerLine = channels * width

            # Convert NumPy array to QImage
            qimage = QImage(png_image_array.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)

            self.ms.update_image.emit(pixmap, "input")
            self.ms.update_image.emit(pixmap, "current")
            self.ms.update_tab_status.emit("input")

            log_text = f"Image uploaded and processed! \n image resized to: {testsize}x{testsize}"
            self.ms.update_text.emit(log_text, "log", "none")

    def segment(self):

        if self.polyp is None:
            QMessageBox.warning(None, 'Warning', 'Please UPLOAD an image before segmentation!')
            return

        testsize = self.testsize
        model = self.segModel
        png_img = self.polyp.png_img

        image_array = np.array(png_img)

        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        image_resized = cv2.resize(image_array, dsize=(352, 352), interpolation=cv2.INTER_LINEAR)
        image_resized = image_resized / 255.0
        image_resized = image_resized.transpose(2, 0, 1)
        image_resized = torch.from_numpy(image_resized).float().unsqueeze(0)

        model.eval()

        # Start timing
        start_time = time.time()

        # Forward pass on the CPU
        with torch.no_grad():
            P1, P2 = model(image_resized)

        # Upsample the demo_result and move it back to CPU
        res = F.interpolate(P1 + P2, size=(352, 352), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()

        # Resize the result to the target size (640x640) using cv2.resize
        res = cv2.resize(res, dsize=(testsize, testsize), interpolation=cv2.INTER_LINEAR)

        if np.all(res == 0):
            QMessageBox.warning(None, 'Warning', 'No ROI segmented!')
            return

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Convert the mask to an 8-bit format and binarize with a threshold of 128
        mask = (res * 255).astype(np.uint8)

        _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

        self.polyp.set_mask(Image.fromarray(binary_mask))

        # Convert NumPy array to QImage
        height, width = binary_mask.shape
        qimage = QImage(binary_mask.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        self.ms.update_image.emit(pixmap, "mask")

        # Ensure png_img is a NumPy array for OpenCV operations
        if not isinstance(png_img, np.ndarray):
            png_img = np.array(png_img)

        # Apply the mask to the image
        image_masked = cv2.bitwise_and(png_img, png_img, mask=binary_mask)

        # Convert image_masked to QImage
        height, width, channels = image_masked.shape
        bytesPerLine = channels * width
        qimage = QImage(image_masked.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        self.ms.update_image.emit(pixmap, "masked")
        self.ms.update_image.emit(pixmap, "current")
        self.ms.update_tab_status.emit("mask")
        self.ms.update_tab_status.emit("masked")

        log_text = f"Segmentation complete! ({elapsed_time:.4f}s)"
        self.ms.update_text.emit(log_text, "log", "none")

    def evaluate(self):

        if self.polyp is None:
            QMessageBox.warning(None, 'Warning', 'Please UPLOAD an image first!')
            return

        img_name = self.polyp.file_name
        png_img = self.polyp.png_img

        if self.polyp.mask is None:
            QMessageBox.warning(None, 'Warning', 'Please SEGMENT the uploaded image!')
            return

        pred_mask = self.polyp.mask
        testsize = self.testsize

        if self.polyp.enhanced_img is None:
            QMessageBox.warning(None, 'Warning', 'Please CLASSIFY the segmented image!')
            return

        gt_folder = self.demo_gt_path
        gt_path = os.path.join(gt_folder, img_name)

        if not os.path.exists(gt_path):
            QMessageBox.warning(None, 'Warning', 'No ground truth found!')
            return

        gt_mask = Image.open(gt_path).convert('L')

        # Resize ground truth image to match the test size
        gt_mask_resized = gt_mask.resize((testsize, testsize), Image.LANCZOS)

        # Set the ground truth image in the Polyp object
        self.polyp.set_gt_img(gt_mask_resized)

        # Ensure the masks are in the correct format
        pred_mask = np.array(pred_mask)  # Convert PIL Image to numpy array if necessary
        pred_mask = (pred_mask * 255).astype(np.uint8)
        gt_mask_resized = np.array(gt_mask_resized)
        gt_mask_resized = (gt_mask_resized * 255).astype(np.uint8)

        def calculate_metrics(pred_mask, true_mask):
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            true_mask = (true_mask > 0.5).astype(np.uint8)

            intersection = np.sum(pred_mask * true_mask)
            dice = (2.0 * intersection) / (np.sum(pred_mask) + np.sum(true_mask) + 1e-8)
            iou = intersection / (np.sum(pred_mask) + np.sum(true_mask) - intersection + 1e-8)

            return dice, iou

        self.polyp.dice, self.polyp.iou = calculate_metrics(pred_mask, gt_mask_resized)

        # Check classification demo_result
        match = re.search(r'^[^-]*-[^-]*-(\d+)', img_name)
        type = ""
        if match:
            if int(match.group(1)) == 1:
                type = "Hyperplastic"
            elif int(match.group(1)) == 2:
                type = "Adenomatous"
            classification_result = "CORRECT" if type == self.polyp.type else "INCORRECT"
        else:
            classification_result = "Unknown"

        # Generate the demo_result string
        result_string = f"Dice: {self.polyp.dice:.4f}, IoU: {self.polyp.iou:.4f},\nClassification: {classification_result}"

        self.ms.update_text.emit(result_string, "evalResult", "none")

        # Convert png_img to numpy array if it's a PIL Image
        if isinstance(png_img, Image.Image):
            png_img = np.array(png_img)

        # Ensure png_img is in the correct format (BGR for OpenCV)
        if png_img.shape[2] == 3:
            png_img = cv2.cvtColor(png_img, cv2.COLOR_RGB2BGR)

        # Create overlays for the masks
        overlay = png_img.copy()
        overlay[pred_mask > 128] = [0, 0, 255]  # Red color for the predicted mask in BGR
        overlay[gt_mask_resized > 128] = [0, 255, 0]  # Green color for the ground truth mask in BGR

        # Create transparency masks
        alpha = 0.5  # Transparency factor for predicted mask
        beta = 0.3  # Transparency factor for ground truth mask
        combined = cv2.addWeighted(overlay, alpha, png_img, 1 - alpha, 0)
        combined = cv2.addWeighted(combined, beta, overlay, 1 - beta, 0)

        # Find the contours of the masks
        predicted_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gt_contours, _ = cv2.findContours(gt_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(combined, predicted_contours, -1, (0, 0, 255), 2)  # Red color for predicted contours in BGR
        cv2.drawContours(combined, gt_contours, -1, (0, 255, 0), 2)  # Green color for ground truth contours in BGR

        print(f"Ground truth image '{img_name}' loaded and processed")

        # Convert NumPy array to QImage
        combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        height, width, channels = combined.shape
        bytesPerLine = channels * width
        qimage = QImage(combined.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        self.ms.update_image.emit(pixmap, "eval")
        self.ms.update_image.emit(pixmap, "current")
        self.ms.update_tab_status.emit("eval")

        log_text = f"Evaluation complete!"
        self.ms.update_text.emit(log_text, "log", "none")

    def extract_feature(self):

        if self.polyp is None:
            QMessageBox.warning(None, 'Warning', 'Please UPLOAD an image before segmentation!')
            return

        image = self.polyp.png_img

        if self.polyp.mask is None:
            QMessageBox.warning(None, 'Warning', 'Please SEGMENT the uploaded image before classification!')
            return

        mask = self.polyp.mask

        # Start timing
        start_time = time.time()

        # Convert PIL images to NumPy arrays, simulating cv2.imread format
        image = np.array(image)
        mask = np.array(mask.convert('L'))  # Convert mask to grayscale

        extractor = ImageFeatureExtractor(image, mask)
        features, image_enhanced, image_masked = extractor.process()

        rounded_features = {k: round(float(v), 4) for k, v in features.items()}

        # Initialize dictionary to store feature differences
        feature_diff = {}
        # Initialize list to store color labels
        color_labels = []

        # List of features with reversed labeling rules (features having higher values for hyper-plastic polyps)
        reversed_features = [
            "GLCM_energy", "GLCM_homogeneity",
            "Circularity", "BED_cv"
        ]

        # Calculate differences for each feature
        for feature_name, feature_value in rounded_features.items():
            # Construct the attribute names for _1 and _2
            attr_name_1 = f"{feature_name}_Adenomatous"
            attr_name_2 = f"{feature_name}_Hyperplastic"

            # Get the values of feature_name_1 and feature_name_2
            feature_1_value = getattr(self, attr_name_1, 0)
            feature_2_value = getattr(self, attr_name_2, 0)

            # Calculate differences
            diff_1 = feature_value - feature_1_value
            diff_2 = feature_value - feature_2_value

            # Store the differences in the dictionary
            feature_diff[feature_name] = (diff_1, diff_2)

            # Determine the color label based on the differences and reversed rules
            if feature_name in reversed_features:
                if diff_1 > 0 and diff_2 > 0:
                    color_labels.append("green")
                elif (diff_1 < 0 < diff_2) or (diff_2 < 0 < diff_1):
                    color_labels.append("yellow")
                elif diff_1 < 0 and diff_2 < 0:
                    color_labels.append("red")
                else:
                    color_labels.append("yellow")
            else:
                if diff_1 < 0 and diff_2 < 0:
                    color_labels.append("green")
                elif (diff_1 < 0 < diff_2) or (diff_2 < 0 < diff_1):
                    color_labels.append("yellow")
                elif diff_1 > 0 and diff_2 > 0:
                    color_labels.append("red")
                else:
                    color_labels.append("yellow")

        self.polyp.set_features(rounded_features)
        self.polyp.set_enhanced_img(Image.fromarray(image_enhanced))

        height, width = image_enhanced.shape
        bytesPerLine = width
        qimage = QImage(image_enhanced.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        self.ms.update_image.emit(pixmap, "enhanced")
        self.ms.update_image.emit(pixmap, "current")
        self.ms.update_tab_status.emit("enhanced")

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time

        log_text = f"Feature extraction complete! ({elapsed_time:.4f}s)"
        self.ms.update_text.emit(log_text, "log", "none")

        self.ms.update_feature_table.emit(rounded_features)

        self.ms.set_cell_color.emit(color_labels)

    def classify(self):

        if self.polyp.mask is None:
            return

        scaler = self.scaler
        clsModel = self.clsModel
        selector = self.selector

        # Start timing
        start_time = time.time()

        # Get data from polyp attributes
        data = self.polyp.get_polyp_features()

        # Ensure data is a 2D array
        data_2d = np.array(data).reshape(1, -1)

        # Get feature names
        feature_names = scaler.feature_names_in_

        # Create DataFrame
        data_df = pd.DataFrame(data_2d, columns=feature_names)

        # Standardize the data
        scaled_data = scaler.transform(data_df)

        if selector is None:
            selected_data = scaled_data
        else:
            # Select features
            selected_data = selector.transform(scaled_data)

        probabilities = clsModel.predict_proba(selected_data)

        # Get the predicted class and its probability
        predicted_class = np.argmax(probabilities)

        confidence = 100 * probabilities[0][predicted_class]

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time

        log_text = f"Classification complete! ({elapsed_time:.4f}s)"
        self.ms.update_text.emit(log_text, "log", "none")

        if predicted_class == 1:
            self.polyp.type = 'Hyperplastic'
            result_string = f"{self.polyp.type} (Confidence: {confidence:.2f}%)"
            self.ms.update_text.emit(result_string, "clsResult", "green")
        else:
            self.polyp.type = 'Adenomatous'
            result_string = f"{self.polyp.type} (Confidence: {confidence:.2f}%)"
            self.ms.update_text.emit(result_string, "clsResult", "red")

        sorted_importances = self.sorted_importances

        # Create a string with feature importances
        importance_string = "Feature Contributions:\n"
        for feature, importance in sorted_importances:
            importance_string += f"{feature}: {importance:.4f}\n"

        self.ms.add_importance.emit(importance_string)

    def export(self):

        if self.polyp is None:
            QMessageBox.warning(None, 'Warning', 'Please UPLOAD an image before segmentation!')
            return

        export_dir = self.demo_output_dir

        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)

        os.makedirs(export_dir)

        if self.polyp.png_img:
            self.polyp.png_img.save(os.path.join(export_dir, f"{self.polyp.file_name}.png"))
        if self.polyp.mask:
            self.polyp.mask.save(os.path.join(export_dir, f"mask_{self.polyp.file_name}.png"))
        if self.polyp.masked_img:
            self.polyp.masked_img.save(os.path.join(export_dir, f"masked_img_{self.polyp.file_name}.png"))
        if self.polyp.enhanced_img:
            self.polyp.enhanced_img.save(os.path.join(export_dir, f"enhanced_img_{self.polyp.file_name}.png"))
        if self.polyp.gt_img:
            self.polyp.gt_img.save(os.path.join(export_dir, f"gt_img_{self.polyp.file_name}.png"))

        with open(os.path.join(export_dir, "classification_result.txt"), "w") as file:
            file.write(f"The type of {self.polyp.file_name} is {self.polyp.type}")

        log_text = f"Images and file name have been exported to the '{export_dir}' directory!"
        self.ms.update_text.emit(log_text, "log", "none")

    def batch_process(self, input_dir, output_dir):
        testsize = self.testsize
        model = self.segModel
        scaler = self.scaler
        clsModel = self.clsModel
        selector = self.selector

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)
        # Create a subdirectory for masks
        masks_dir = os.path.join(output_dir, 'masks')
        if not os.path.exists(masks_dir):
            os.makedirs(masks_dir)

        # Get all files from the input directory, excluding 'desktop.ini'
        files = [f for f in os.listdir(input_dir)
                 if os.path.isfile(os.path.join(input_dir, f)) and f.lower() != 'desktop.ini']

        # Number of images to process
        num_images = len(files)

        # Initialize the index for progress tracking
        index = 0

        # List to store results
        results = []

        # Log completion
        log_text = f"Batch Processing starts! (total {num_images} images)"
        self.ms.update_text.emit(log_text, "log", "none")

        for file in files:
            # Get the full path of the file
            file_path = os.path.join(input_dir, file)

            self.polyp = Polyp()

            if file.lower().endswith('.dcm'):
                # Read DICOM file
                ds = pydicom.dcmread(file_path)
                img_array = ds.pixel_array
                img = Image.fromarray(img_array)
            else:
                # Read other image files
                img = Image.open(file_path)

            # Convert to PNG format
            png_img = img.convert("RGB")

            # Resize the image
            png_img = png_img.resize((testsize, testsize), Image.LANCZOS)

            self.polyp.set_file_name(os.path.basename(file_path))
            self.polyp.set_png_img(png_img)

            image_array = np.array(png_img)

            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]

            if self.mode == "classifier only" or self.mode == "feature only":
                mask_path = os.path.join(self.batch_gt_mask_dir, self.polyp.file_name)
                if os.path.exists(mask_path):
                    binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                else:
                    raise FileNotFoundError(f"Mask file {self.polyp.file_name} not found in gt_mask folder.")

            else:
                image_resized = cv2.resize(image_array, dsize=(352, 352), interpolation=cv2.INTER_LINEAR)
                image_resized = image_resized / 255.0
                image_resized = image_resized.transpose(2, 0, 1)
                image_resized = torch.from_numpy(image_resized).float().unsqueeze(0)

                model.eval()

                # Forward pass on the CPU
                with torch.no_grad():
                    P1, P2 = model(image_resized)

                # Upsample the demo_result and move it back to CPU
                res = F.interpolate(P1 + P2, size=(352, 352), mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()

                # Resize the result to the target size (640x640) using cv2.resize
                res = cv2.resize(res, dsize=(testsize, testsize), interpolation=cv2.INTER_LINEAR)

                # Convert the mask to an 8-bit format and binarize with a threshold of 128
                mask = (res * 255).astype(np.uint8)
                _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

                # Save the mask to the masks directory
                mask_img = Image.fromarray(binary_mask)
                mask_file_path = os.path.join(masks_dir, f"{os.path.splitext(file)[0]}.png")
                mask_img.save(mask_file_path)

                self.polyp.set_mask(Image.fromarray(mask))

            extractor = ImageFeatureExtractor(image_array, binary_mask)
            features, image_enhanced, image_masked = extractor.process()

            self.polyp.set_features(features)
            self.polyp.set_enhanced_img(Image.fromarray(image_enhanced))

            # Get data from polyp attributes
            data = self.polyp.get_polyp_features()

            if self.mode == "feature only":

                self.polyp.type = 'N/A'

            else:

                # Ensure data is a 2D array
                data_2d = np.array(data).reshape(1, -1)

                # Get feature names
                feature_names = scaler.feature_names_in_

                # Create DataFrame
                data_df = pd.DataFrame(data_2d, columns=feature_names)

                # Standardize the data
                scaled_data = scaler.transform(data_df)

                if selector is None:
                    selected_data = scaled_data
                else:
                    # Select features
                    selected_data = selector.transform(scaled_data)

                # Make predictions with probabilities
                probabilities = clsModel.predict_proba(selected_data)

                # Get the predicted class and its probability
                predicted_class = np.argmax(probabilities)

                if predicted_class == 1:
                    self.polyp.type = 'Hyperplastic'
                elif predicted_class == 0:
                    self.polyp.type = 'Adenomatous'

            # Get data from polyp attributes
            data = self.polyp.to_dict()

            try:
                # Append the data to the results list
                results.append(data)
            except Exception as e:
                print(f"Failed to append data: {e}")

            # Update progress bar
            self.ms.update_processBar.emit((index + 1) * 100 // num_images)

            # Increment the index
            index += 1

        # Create a DataFrame to store all results
        results_df = pd.DataFrame(results)

        # Save the results DataFrame to an Excel file
        excel_path = os.path.join(output_dir, self.export_filename)
        results_df.to_excel(excel_path, index=False)

        # Log completion
        log_text = (
            f"--------------------------------------\n Batch Processing complete! Processed {num_images} images.\n Predicted masks and"
            f"classification report exported to the '{output_dir}' directory!")
        self.ms.update_text.emit(log_text, "log", "none")
