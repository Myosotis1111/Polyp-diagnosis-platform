from PIL import Image


class Polyp:

    def __init__(self, type="", iou=0, dice=0, CCS_mean=0, CCS_median=0, CCS_var=0, CCS_cv=0,
                 Edge_Intensity=0, Edge_Density=0, BED_mean=0, BED_cv=0, BED_median=0, BED_var=0, circularity=0,
                 HD_mean=0, HD_cv=0, HD_median=0, HD_var=0, TVS_mean=0, TVS_var=0, TVS_median=0,
                 glcm_contrast=0, glcm_dissimilarity=0,
                 glcm_homogeneity=0, glcm_energy=0, glcm_correlation=0,
                 ROI_proportion=0, ROI_var=0, ROI_evar=0,
                 png_img=None, mask=None, masked_img=None, enhanced_img=None, gt_img=None, file_name=""):
        self.type = type
        self.iou = iou
        self.dice = dice
        self.CCS_mean = CCS_mean
        self.CCS_median = CCS_median
        self.CCS_var = CCS_var
        self.CCS_cv = CCS_cv
        self.Edge_Intensity = Edge_Intensity
        self.Edge_Density = Edge_Density
        self.BED_mean = BED_mean
        self.BED_cv = BED_cv
        self.BED_median = BED_median
        self.BED_var = BED_var
        self.Circularity = circularity
        self.HD_mean = HD_mean
        self.HD_cv = HD_cv
        self.HD_median = HD_median
        self.HD_var = HD_var
        self.TVS_mean = TVS_mean
        self.TVS_var = TVS_var
        self.TVS_median = TVS_median
        self.GLCM_contrast = glcm_contrast
        self.GLCM_dissimilarity = glcm_dissimilarity
        self.GLCM_homogeneity = glcm_homogeneity
        self.GLCM_energy = glcm_energy
        self.GLCM_correlation = glcm_correlation
        self.ROI_proportion = ROI_proportion
        self.ROI_var = ROI_var
        self.ROI_evar = ROI_evar

        # New image attributes
        self.png_img = png_img
        self.mask = mask
        self.masked_img = masked_img
        self.enhanced_img = enhanced_img
        self.gt_img = gt_img

        # New file_name attribute
        self.file_name = file_name

    def to_dict(self):
        return {
            'file_name': self.file_name,
            'Edge Intensity': self.Edge_Intensity,
            'CCS_mean': self.CCS_mean,
            'CCS_median': self.CCS_median,
            'CCS_var': self.CCS_var,
            'CCS_cv': self.CCS_cv,
            'Edge Density': self.Edge_Density,
            'HD_mean': self.HD_mean,
            'HD_median': self.HD_median,
            'HD_var': self.HD_var,
            'HD_cv': self.HD_cv,
            'TVS_mean': self.TVS_mean,
            'TVS_median': self.TVS_median,
            'TVS_var': self.TVS_var,
            'GLCM_contrast': self.GLCM_contrast,
            'GLCM_dissimilarity': self.GLCM_dissimilarity,
            'GLCM_homogeneity': self.GLCM_homogeneity,
            'GLCM_energy': self.GLCM_energy,
            'GLCM_correlation': self.GLCM_correlation,
            'ROI_var': self.ROI_var,
            'ROI_evar': self.ROI_evar,
            'BED_mean': self.BED_mean,
            'BED_median': self.BED_median,
            'BED_var': self.BED_var,
            'BED_cv': self.BED_cv,
            'Circularity': self.Circularity,
            'ROI_proportion': self.ROI_proportion,
            'Classification type': self.type,
        }

    def set_png_img(self, img):
        if isinstance(img, Image.Image):
            self.png_img = img
        else:
            raise ValueError("png_img must be a PIL Image object")

    def set_mask(self, img):
        if isinstance(img, Image.Image):
            self.mask = img
        else:
            raise ValueError("mask must be a PIL Image object")

    def set_masked_img(self, img):
        if isinstance(img, Image.Image):
            self.masked_img = img
        else:
            raise ValueError("masked_img must be a PIL Image object")

    def set_enhanced_img(self, img):
        if isinstance(img, Image.Image):
            self.enhanced_img = img
        else:
            raise ValueError("enhanced_img must be a PIL Image object")

    def set_gt_img(self, img):
        if isinstance(img, Image.Image):
            self.gt_img = img
        else:
            raise ValueError("gt_img must be a PIL Image object")

    def set_file_name(self, name):
        if isinstance(name, str):
            self.file_name = name
        else:
            raise ValueError("file_name must be a string")

    def set_features(self, features):

        # Assign extracted features to the corresponding attributes
        self.CCS_mean = features.get('CCS_mean', 0)
        self.CCS_median = features.get('CCS_median', 0)
        self.CCS_var = features.get('CCS_var', 0)
        self.CCS_cv = features.get('CCS_cv', 0)
        self.Edge_Intensity = features.get('Edge Intensity', 0)
        self.Edge_Density = features.get('Edge Density', 0)
        self.BED_mean = features.get('BED_mean', 0)
        self.BED_cv = features.get('BED_cv', 0)
        self.BED_median = features.get('BED_median', 0)
        self.BED_var = features.get('BED_var', 0)
        self.Circularity = features.get('Circularity', 0)
        self.HD_mean = features.get('HD_mean', 0)
        self.HD_cv = features.get('HD_cv', 0)
        self.HD_median = features.get('HD_median', 0)
        self.HD_var = features.get('HD_var', 0)
        self.TVS_mean = features.get('TVS_mean', 0)
        self.TVS_var = features.get('TVS_var', 0)
        self.TVS_median = features.get('TVS_median', 0)
        self.GLCM_contrast = features.get('GLCM_contrast', 0)
        self.GLCM_dissimilarity = features.get('GLCM_dissimilarity', 0)
        self.GLCM_homogeneity = features.get('GLCM_homogeneity', 0)
        self.GLCM_energy = features.get('GLCM_energy', 0)
        self.GLCM_correlation = features.get('GLCM_correlation', 0)
        self.ROI_var = features.get('ROI_var', 0)
        self.ROI_evar = features.get('ROI_evar', 0)
        self.ROI_proportion = features.get('ROI_proportion', 0)

    def get_polyp_features(self):
        features = [
            self.Edge_Intensity,
            self.CCS_mean,
            self.CCS_median,
            self.CCS_var,
            self.CCS_cv,
            self.Edge_Density,
            self.HD_mean,
            self.HD_median,
            self.HD_var,
            self.HD_cv,
            self.TVS_mean,
            self.TVS_median,
            self.TVS_var,
            self.GLCM_contrast,
            self.GLCM_dissimilarity,
            self.GLCM_homogeneity,
            self.GLCM_energy,
            self.GLCM_correlation,
            self.ROI_var,
            self.ROI_evar,
            self.BED_mean,
            self.BED_median,
            self.BED_var,
            self.BED_cv,
            self.Circularity,
            self.ROI_proportion

        ]
        return [features]
