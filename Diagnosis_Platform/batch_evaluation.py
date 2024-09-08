import os
import re
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_iou_and_dice(mask, gt):
    """
    Calculate the Intersection over Union (IoU) and Dice coefficient between the mask and ground truth.
    """
    intersection = np.logical_and(mask, gt).sum()
    union = np.logical_or(mask, gt).sum()
    iou = intersection / union if union != 0 else 0
    dice = (2 * intersection) / (mask.sum() + gt.sum()) if (mask.sum() + gt.sum()) != 0 else 0
    return iou, dice


def main():
    mask_dir = 'batch_io/batch_result/masks'

    gt_dir = 'batch_io/batch_gt_mask'
    categories = {
        "Hyperplastic": {'iou': [], 'dice': []},
        "Adenomatous": {'iou': [], 'dice': []}
    }

    # Read the Excel file containing classification results
    classification_result_path = 'batch_io/batch_result/test_features.xlsx'

    # # activate below if you want to evaluate classification model only
    # mask_dir = 'batch_io/batch_gt_mask'  # When only classifier is tested

    classification_df = pd.read_excel(classification_result_path)

    y_true = []
    y_pred = []
    wrong_predictions = []

    for mask_filename in os.listdir(mask_dir):
        if mask_filename.endswith('.png'):
            # Use regular expressions to extract the category from the filename
            match = re.search(r'^[^-]*-[^-]*-(\d+)', mask_filename)
            category = ""
            if match:
                if int(match.group(1)) == 1:
                    category = "Hyperplastic"
                elif int(match.group(1)) == 2:
                    category = "Adenomatous"
                else:
                    print(f"Unrecognized category number {match.group(1)} in filename {mask_filename}")
                    continue
            else:
                raise ValueError("The provided file_name does not match the expected format.")

            # Get the predicted category from the Excel file
            excel_row = classification_df[classification_df.iloc[:, 0] == mask_filename]
            if excel_row.empty:
                print(f"Filename {mask_filename} not found in classification results")
                continue

            predicted_category = excel_row.iloc[0, -1]
            y_true.append(category)
            y_pred.append(predicted_category)

            # Read the mask and corresponding ground truth
            mask_path = os.path.join(mask_dir, mask_filename)
            gt_path = os.path.join(gt_dir, mask_filename)
            if not os.path.exists(gt_path):
                print(f"Ground truth for {mask_filename} does not exist")
                continue

            mask = Image.open(mask_path).convert('L')
            gt = Image.open(gt_path).convert('L')

            # Resize the mask to match the size of the ground truth
            mask = mask.resize(gt.size, Image.Resampling.LANCZOS)
            mask = np.array(mask) > 0
            gt = np.array(gt) > 0

            # Calculate IoU and Dice
            iou, dice = calculate_iou_and_dice(mask, gt)

            # Store the metrics in the corresponding category
            categories[category]['iou'].append(iou)
            categories[category]['dice'].append(dice)

            # Check if prediction is wrong and collect details if it is
            if predicted_category != category:
                wrong_predictions.append({
                    'filename': mask_filename,
                    'true_category': category,
                    'predicted_category': predicted_category,
                    'iou': iou,
                    'dice': dice
                })

    # Print details of wrong predictions in a condensed format
    if wrong_predictions:
        print("\nWrong Predictions:")
        for item in wrong_predictions:
            print(
                f"{item['filename']}: True={item['true_category']}, Pred={item['predicted_category']}, IoU={item['iou']:.4f}, Dice={item['dice']:.4f}")
    else:
        print("\nNo wrong predictions.")

    # Calculate mIoU and mDice for each category
    overall_iou = []
    overall_dice = []

    print("\n----------------------evaluation report-----------------------\n")

    print("------------------Segmentation performance------------------")
    for category, metrics in categories.items():
        if metrics['iou']:
            mIoU = np.mean(metrics['iou'])
            mDice = np.mean(metrics['dice'])
            overall_iou.extend(metrics['iou'])
            overall_dice.extend(metrics['dice'])

            print(f"Category {category}: mIoU = {mIoU:.4f}, mDice = {mDice:.4f}")
        else:
            print(f"Category {category}: No data to calculate mIoU and mDice")

    # Calculate overall mIoU and mDice
    if overall_iou:
        mIoU_all = np.mean(overall_iou)
        mDice_all = np.mean(overall_dice)
        print(f"Overall: mIoU = {mIoU_all:.4f}, mDice = {mDice_all:.4f}")
    else:
        print("Overall: No data to calculate mIoU and mDice\n")

    # Calculate F1 score, recall, and accuracy
    labels = ["Hyperplastic", "Adenomatous"]
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Calculate F1 Score, Recall, and Accuracy
    overall_f1 = f1_score(y_true, y_pred, labels=labels, average='weighted')
    overall_recall = recall_score(y_true, y_pred, labels=labels, average='weighted')
    overall_accuracy = accuracy_score(y_true, y_pred)

    # Calculate per-class F1 Score, Recall, and Accuracy
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None)

    # Calculate per-class accuracy
    per_class_accuracy = np.diag(cm) / np.sum(cm, axis=1)

    print("------------------Classification performance------------------")

    # Print per-class metrics
    for label, f1, rec, acc in zip(labels, per_class_f1, per_class_recall, per_class_accuracy):
        print(f"Class '{label}' - F1 Score: {f1:.4f}, Recall: {rec:.4f}, Accuracy: {acc:.4f}")

    # Print overall metrics
    print(f"\nOverall F1 Score: {overall_f1:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    plot_metric_distributions(categories)


def plot_metric_distributions(categories):
    """
    Plot the distribution of IoU and Dice coefficients for each category in the same figure.
    Each category will have two box plots side-by-side: one for IoU and one for Dice.
    """
    # Convert data to DataFrame for easier plotting
    data = []
    for category, metrics in categories.items():
        for iou, dice in zip(metrics['iou'], metrics['dice']):
            data.append({'Category': category, 'Metric': 'IoU', 'Value': iou})
            data.append({'Category': category, 'Metric': 'Dice', 'Value': dice})
    df = pd.DataFrame(data)

    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))

    # Plot the boxplot
    sns.boxplot(x='Category', y='Value', hue='Metric', data=df,
                width=0.6)

    # Title and labels
    plt.title('IoU and Dice Distribution by Category')
    plt.ylabel('Score')
    plt.xlabel('Category')
    plt.ylim(0, 1)  # Both IoU and Dice values are between 0 and 1
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
