import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from skimage.feature import graycomatrix, graycoprops


def sharpen_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_sharpened = cv2.Laplacian(img_blurred, cv2.CV_8U, ksize=5)
    return img_sharpened


def binary_threshold(image, multiplier=1.2):
    img = Image.fromarray(image)
    pixels = list(img.getdata())
    pixels = [p for p in pixels if 1 <= p <= 254]
    if len(pixels) > 0:
        threshold = int(multiplier * sum(pixels) / len(pixels))
    else:
        threshold = 127
    return threshold


def fit_ellipse(contour):
    if len(contour) < 5:
        return None
    try:
        ellipse = cv2.fitEllipse(contour)
        return ellipse
    except:
        return None


def calibrate_image(image):
    def find_nonzero_regions(masked_image):
        contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nonzero_regions = []
        for contour in contours:
            ellipse = fit_ellipse(contour)
            if ellipse is None:
                continue
            (center_x, center_y), (minor_axis, major_axis), angle = ellipse
            nonzero_regions.append({
                'center': (int(center_x), int(center_y)),
                'ellipse': ellipse
            })
        return nonzero_regions

    def rotate_image(image, angle, center):
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        return rotated_image

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nonzero_regions = find_nonzero_regions(gray_image)
    if len(nonzero_regions) > 1:
        rotated_image = image
        ellipse = None
    else:
        ellipse = nonzero_regions[0]['ellipse']
        (center_x, center_y), (minor_axis, major_axis), angle = ellipse
        if major_axis > minor_axis:
            angle = 90 - angle
        rotated_image = rotate_image(image.copy(), angle, (center_x, center_y))
    return rotated_image, ellipse


def calculate_size(mask):
    total_pixels = mask.size
    white_pixels = np.sum(mask == 255)
    ROI_proportion = white_pixels / total_pixels
    return ROI_proportion


class ImageFeatureExtractor:
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask
        self.masked_image = None

    """
    Feature Group 1: Edge-based features
    """

    def calculate_edge_features(self):
        img_sharpened = sharpen_image(self.masked_image)
        threshold = binary_threshold(img_sharpened, multiplier=1.2)
        _, binarized_img = cv2.threshold(img_sharpened, threshold, 255, cv2.THRESH_BINARY)
        img_denoised = cv2.medianBlur(binarized_img, 3)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_denoised, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        indices = np.arange(1, num_labels)
        sorted_indices = sorted(indices, key=lambda k: areas[k - 1])
        sorted_indices = [idx for idx in sorted_indices if areas[idx - 1] >= 4]
        num_to_keep = len(sorted_indices) // 2
        sorted_indices = sorted_indices[num_to_keep:]
        percent_to_remove = 0.02
        num_to_remove = max(3, int(np.ceil(len(sorted_indices) * percent_to_remove)))
        sorted_indices = sorted_indices[:-num_to_remove]
        removed_indices = sorted_indices[-num_to_remove:]
        removed_area_sum = sum(areas[idx - 1] for idx in removed_indices)
        remaining_areas = [areas[idx - 1] for idx in sorted_indices]
        mean_area = np.mean(remaining_areas)
        median_area = np.median(remaining_areas)
        var_area = np.var(remaining_areas)
        std_area = np.std(remaining_areas)
        cv_area = std_area / mean_area if mean_area != 0 else float('inf')
        roi1 = cv2.bitwise_and(img_sharpened, img_sharpened, mask=self.mask)
        non_black_pixels = roi1[roi1 > 0]
        roi2 = cv2.bitwise_and(img_denoised, img_denoised, mask=self.mask)
        white_pixels = roi2[roi1 > 0]
        avg_brightness = np.mean(non_black_pixels) if len(non_black_pixels) > 0 else 0
        white_ratio = (len(white_pixels) - removed_area_sum) / np.count_nonzero(self.mask) if np.count_nonzero(
            self.mask) > 0 else 0
        return {
            'Edge Intensity': avg_brightness,
            'CCS_mean': mean_area,
            'CCS_median': median_area,
            'CCS_var': var_area,
            'CCS_cv': cv_area,
            'Edge Density': white_ratio
        }, img_denoised

    """
    Feature Group 2: Histogram-based features
    """

    def calculate_histogram_features(self, n=4):
        def split_image(image, n):
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return []
            x, y, w, h = cv2.boundingRect(contours[0])
            cell_width = w // n
            cell_height = h // n
            sub_images = []
            for i in range(n):
                for j in range(n):
                    x_start = x + j * cell_width
                    y_start = y + i * cell_height
                    sub_image = image[y_start:y_start + cell_height, x_start:x_start + cell_width]
                    sub_images.append(sub_image)
            return sub_images

        def calculate_histogram(image):
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = np.where((gray_image == 0) | (gray_image > 250), 0, 1).astype(np.uint8)
            hist = cv2.calcHist([gray_image], [0], mask, [251], [1, 251])
            hist = cv2.normalize(hist, hist).flatten()
            return hist

        def compare_histograms(hist1, hist2):
            return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

        rotated_image, _ = calibrate_image(self.masked_image)
        sub_images = split_image(rotated_image, n)
        if len(sub_images) == 0:
            return {
                'HD_mean': 0,
                'HD_median': 0,
                'HD_var': 0,
                'HD_cv': 0
            }

        else:
            histograms = []
            for sub_image in sub_images:
                gray_sub_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
                mask = (gray_sub_image == 0) | (gray_sub_image > 250)
                if np.sum(mask) / mask.size > 0.8:
                    continue

                # Calculate histogram only if the sub-image passes the filter
                histograms.append(calculate_histogram(sub_image))

            distances = []
            for i in range(len(histograms)):
                for j in range(i + 1, len(histograms)):
                    distance = compare_histograms(histograms[i], histograms[j])
                    distances.append(distance)
            mean_distance = np.mean(distances)
            median_distance = np.median(distances)
            var_distance = np.var(distances)
            cv_distance = np.std(distances) / mean_distance
            return {
                'HD_mean': mean_distance,
                'HD_median': median_distance,
                'HD_var': var_distance,
                'HD_cv': cv_distance
            }

    """
    Feature Group 3: Transition-based features
    """

    def calculate_transition_features(self, line_length=20, region_size=31):
        image = self.image
        mask = self.mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundary_image = np.zeros_like(image)
        cv2.drawContours(boundary_image, contours, -1, (255, 0, 0), 2)

        kernel_inner = np.ones((region_size, region_size), np.uint8)
        kernel_outer = np.ones((region_size, region_size), np.uint8)

        inner_mask = cv2.erode(mask, kernel_inner, iterations=1)
        outer_mask = cv2.dilate(mask, kernel_outer, iterations=1)

        boundary_neighbors = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))
        neighbor_image = cv2.bitwise_and(image, image, mask=boundary_neighbors)
        contour_image = neighbor_image.copy()

        cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)
        contour = contours[0]

        num_points = len(contour)

        step = len(contour) // num_points
        selected_points = [contour[i][0] for i in range(0, len(contour), step)][:num_points]

        all_pixels_perpendicular = []
        max_pixels = 0

        for idx, point_on_boundary in enumerate(selected_points):
            point_idx = np.where((contour == point_on_boundary).all(axis=2))[0][0]
            neighbors = 5
            points = []
            for i in range(point_idx - neighbors, point_idx + neighbors + 1):
                idx = i % len(contour)
                points.append(contour[idx][0])
            points = np.array(points)

            x = points[:, 0]
            y = points[:, 1]
            coefficients = np.polyfit(x, y, 1)
            tangent_direction = np.arctan(coefficients[0])
            perpendicular_direction = tangent_direction + np.pi / 2

            for i in range(-line_length, line_length + 1):
                start_point = (int(point_on_boundary[0] + i * np.cos(perpendicular_direction)),
                               int(point_on_boundary[1] + i * np.sin(perpendicular_direction)))
                end_point = (int(point_on_boundary[0] + (i + 1) * np.cos(perpendicular_direction)),
                             int(point_on_boundary[1] + (i + 1) * np.sin(perpendicular_direction)))
                cv2.line(contour_image, start_point, end_point, (0, 255, 0), 1)

            pixels_perpendicular = []
            for i in range(-line_length, line_length + 1):
                new_x = int(point_on_boundary[0] + i * np.cos(perpendicular_direction))
                new_y = int(point_on_boundary[1] + i * np.sin(perpendicular_direction))
                if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]:
                    pixel_value = image[new_y, new_x]
                    if not np.array_equal(pixel_value, [0, 0, 0]):
                        pixels_perpendicular.append(pixel_value)

            max_pixels = max(max_pixels, len(pixels_perpendicular))
            all_pixels_perpendicular.append(pixels_perpendicular)

        variability_scores = []
        for idx, pixels_perpendicular in enumerate(all_pixels_perpendicular):
            if len(pixels_perpendicular) < 2:
                variability_scores.append(0.0)
                continue
            pixels_array = np.array(pixels_perpendicular)
            color_std = np.std(pixels_array, axis=0)
            if np.sum(color_std) == 0:
                variability_score = 1.0
            else:
                variability_score = np.sum(color_std)
            variability_scores.append(variability_score)

        sorted_scores = sorted(variability_scores)
        half_idx = len(sorted_scores) // 2
        variability_scores = sorted_scores[:half_idx]

        if len(variability_scores) == 0:
            return {'TVS_mean': 0, 'TVS_median': 0, 'TVS_var': 0}

        variability_scores = np.array(variability_scores)
        TVS_mean = np.mean(variability_scores)
        TVS_median = np.median(variability_scores)
        TVS_var = np.var(variability_scores)

        transition_features = {
            'TVS_mean': TVS_mean,
            'TVS_median': TVS_median,
            'TVS_var': TVS_var,
        }

        return transition_features

    """
    Feature Group 4: GLCM-based features
    """

    def calculate_glcm_features(self):
        gray_image = cv2.cvtColor(self.masked_image, cv2.COLOR_BGR2GRAY)
        distances = [1]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True,
                            normed=True)

        # 计算各个特征
        features = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        feature_values = {feature: graycoprops(glcm, feature).mean() for feature in features}

        return {
            'GLCM_contrast': feature_values['contrast'],
            'GLCM_dissimilarity': feature_values['dissimilarity'],
            'GLCM_homogeneity': feature_values['homogeneity'],
            'GLCM_energy': feature_values['energy'],
            'GLCM_correlation': feature_values['correlation'],
        }

    """
    Feature Group 5: Morphology features
    """

    def calculate_morphology_color(self):
        image = self.masked_image

        # Convert image to grayscale if it is a color image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get all non-zero pixels (excluding black pixels)
        non_zero_pixels = image[image > 0]

        # Calculate Pixel Intensity Variance (VAR)
        var = np.var(non_zero_pixels)

        # Calculate Energy Variance (EVAR) based on squared pixel values
        evar = np.var(non_zero_pixels ** 2)

        return {
            'ROI_var': var,
            'ROI_evar': evar
        }

    def calculate_morphology_BED(self):
        def extract_edges(image):
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours

        def calculate_min_distances(contour, ellipse_contour):
            contour_points = contour.reshape(-1, 2)
            ellipse_points = ellipse_contour.reshape(-1, 2)
            num_points = len(contour)
            selected_contour_points = contour_points[
                np.random.choice(contour_points.shape[0], num_points, replace=False)]
            distances = cdist(selected_contour_points, ellipse_points, 'euclidean')
            min_distances = np.min(distances, axis=1)
            return min_distances

        def calculate_cdist(contour, ellipse):
            ellipse_contour = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),
                                               (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                                               int(ellipse[2]), 0, 360, 1)
            min_distances = calculate_min_distances(contour, ellipse_contour)
            return min_distances

        def calculate_circularity(ellipse):
            _, (major_axis, minor_axis), _ = ellipse
            if minor_axis == 0:
                return np.nan
            return major_axis / minor_axis

        contours = extract_edges(self.mask)
        if len(contours) == 0:
            return {
                'BED_mean': 0,
                'BED_median': 0,
                'BED_var': 0,
                'BED_cv': 0,
                'Circularity': 0
            }

        distances = []
        circularities = []
        for contour in contours:
            ellipse = fit_ellipse(contour)
            if ellipse is not None:
                distance = calculate_cdist(contour, ellipse)
                distances.append(distance)
                circularity = calculate_circularity(ellipse)
                circularities.append(circularity)

        if len(distances) == 0 or len(circularities) == 0:
            return {
                'BED_mean': 0,
                'BED_median': 0,
                'BED_var': 0,
                'BED_cv': 0,
                'Circularity': 0
            }

        mean_distance = np.mean(distances)
        median_distance = np.median(distances)
        var_distance = np.var(distances)
        cv_distance = np.std(distances) / mean_distance
        circularity = np.mean(circularities)
        return {
            'BED_mean': mean_distance,
            'BED_median': median_distance,
            'BED_var': var_distance,
            'BED_cv': cv_distance,
            'Circularity': circularity
        }

    """
    Feature Extraction
    """

    def preprocess(self):
        # Find all connected components in the mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.mask, connectivity=8)

        # Find the largest connected component (excluding the background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        # Create a new mask with only the largest connected component
        largest_component_mask = np.zeros_like(self.mask)
        largest_component_mask[labels == largest_label] = 255
        self.mask = largest_component_mask

        # Apply the new mask to the image
        self.masked_image = cv2.bitwise_and(self.image, self.image, mask=largest_component_mask)

        return self.mask, self.masked_image

    def process(self):
        mask, img_masked = self.preprocess()

        # Initialize feature dictionaries
        edge_features = {'Edge Intensity': 0, 'CCS_mean': 0, 'CCS_median': 0, 'CCS_var': 0, 'CCS_cv': 0,
                         'Edge Density': 0}
        histogram_features = {'HD_mean': 0, 'HD_median': 0, 'HD_var': 0, 'HD_cv': 0}
        transition_features = {'TVS_mean': 0, 'TVS_median': 0, 'TVS_var': 0}
        glcm_features = {'GLCM_contrast': 0, 'GLCM_dissimilarity': 0, 'GLCM_homogeneity': 0, 'GLCM_energy': 0,
                         'GLCM_correlation': 0}
        morphology_features_color = {'ROI_var': 0, "ROI_evar": 0}
        morphology_features_BED = {'BED_mean': 0, 'BED_median': 0, 'BED_var': 0, 'BED_cv': 0, 'Circularity': 0}
        ROI_proportion = 0

        # Attempt to calculate dot line score
        try:
            edge_features, img_enhanced = self.calculate_edge_features()
            edge_features = {k: (v if not np.isnan(v) else 0) for k, v in edge_features.items()}
        except:
            img_enhanced = img_masked

        try:
            morphology_features_BED = self.calculate_morphology_BED()
            morphology_features_BED = {k: (v if not np.isnan(v) else 0) for k, v in morphology_features_BED.items()}
        except:
            pass

        try:
            histogram_features = self.calculate_histogram_features()
            histogram_features = {k: (v if not np.isnan(v) else 0) for k, v in histogram_features.items()}
        except:
            pass

        try:
            glcm_features = self.calculate_glcm_features()
            glcm_features = {k: (v if not np.isnan(v) else 0) for k, v in glcm_features.items()}
        except:
            pass

        try:
            transition_features = self.calculate_transition_features()
            transition_features = {k: (v if not np.isnan(v) else 0) for k, v in transition_features.items()}
        except:
            pass

        try:
            morphology_features_color = self.calculate_morphology_color()
            morphology_features_color = {k: (v if not np.isnan(v) else 0) for k, v in morphology_features_color.items()}
        except:
            pass

        try:
            ROI_proportion = calculate_size(self.mask)
            ROI_proportion = ROI_proportion if not np.isnan(ROI_proportion) else 0
        except:
            pass

        # 26 features
        features = {
            **edge_features,
            **histogram_features,
            **transition_features,
            **glcm_features,
            **morphology_features_color,
            **morphology_features_BED,
            'ROI_proportion': ROI_proportion,
        }

        return features, img_enhanced, img_masked
