import numpy as np
import math
import cv2
from scipy.spatial import distance
import mahotas
from scipy.stats import entropy as scipy_entropy


def compute_entropy(tumor_pixels):
    """
    Compute Shannon entropy of a 1D array of pixel intensities (0-255).
    """
    if tumor_pixels.size == 0:
        return 0.0
    hist, _ = np.histogram(tumor_pixels, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]  # remove zeros to avoid log(0)
    return scipy_entropy(hist)


def compute_tumor_features(mask, image, voxel_size=(0.1, 0.1)):
    """
    Compute geometric, intensity, and texture features for a segmented tumor (2D).

    Parameters:
        mask : 2D binary array (0-background, >0 tumor)
        image : 2D grayscale MRI image corresponding to mask (0-255)
        voxel_size : tuple (x, y) of voxel dimensions in cm (default: 0.1 cm = 1 mm)

    Returns:
        features : dict with computed tumor features
    """
    mask_bin = (mask > 0).astype(np.uint8)
    voxel_size = np.array(voxel_size)

    # ---- Area (2D volume) ----
    num_voxels = np.sum(mask_bin)
    area = num_voxels * np.prod(voxel_size)  # cm²

    # ---- Bounding Box and Max Diameter ----
    coords = np.argwhere(mask_bin)
    if coords.size == 0:
        bbox_size = np.zeros(2)
        max_diameter = 0.0
        perimeter = 0.0
    else:
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        bbox_size = (max_coords - min_coords + 1) * voxel_size

        if len(coords) > 1:
            max_diameter = np.max(distance.cdist(coords * voxel_size, coords * voxel_size, 'euclidean'))
        else:
            max_diameter = 0.0

        # ---- Perimeter ----
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = 0.0
        if contours:
            perimeters = [cv2.arcLength(cnt, True) for cnt in contours]
            perimeter = np.sum(perimeters) * voxel_size[0]  # cm

    # ---- Shape features (2D) ----
    if perimeter > 0 and area > 0:
        compactness = 4 * math.pi * area / (perimeter ** 2)   # closer to 1 = rounder
        irregularity = perimeter / (2 * math.sqrt(math.pi * area))
    else:
        compactness = irregularity = 0.0

    # ---- Intensity features ----
    tumor_pixels = image[mask_bin > 0]
    if tumor_pixels.size > 0:
        mean_intensity = tumor_pixels.mean()
        std_intensity = tumor_pixels.std()
        min_intensity = tumor_pixels.min()
        max_intensity = tumor_pixels.max()
    else:
        mean_intensity = std_intensity = min_intensity = max_intensity = 0.0

    # ---- Texture features (Haralick + Entropy) ----
    image_uint8 = image.astype(np.uint8)
    if tumor_pixels.size > 0:
        # Crop tumor region to minimize background influence
        ys, xs = np.nonzero(mask_bin)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        tumor_crop = image_uint8[y_min:y_max + 1, x_min:x_max + 1]

        haralick_features = mahotas.features.haralick(tumor_crop, return_mean=True)
        contrast = haralick_features[1]  # contrast
        homogeneity = haralick_features[4]  # homogeneity
        entropy_value = compute_entropy(tumor_pixels)
    else:
        contrast = homogeneity = entropy_value = 0.0

    # ---- Combine results ----
    features = {
        'Area (cm^2)': area,
        'Bounding Box (cm)': bbox_size,
        'Max Diameter (cm)': max_diameter,
        'Perimeter (cm)': perimeter,
        'Compactness': compactness,
        'Irregularity': irregularity,
        'Mean Intensity': mean_intensity,
        'Std Intensity': std_intensity,
        'Min Intensity': min_intensity,
        'Max Intensity': max_intensity,
        'Contrast': contrast,
        'Homogeneity': homogeneity,
        'Entropy': entropy_value
    }

    return features


def features_note():
    """
    Returns explanations for each tumor feature for clinical interpretation.
    """
    feature_explanations = {
        # Geometric features
        'Area (cm^2)': "Total tumor cross-sectional area (2D), computed from mask and voxel size.",
        'Bounding Box (cm)': "Width and height of the smallest box enclosing the tumor.",
        'Max Diameter (cm)': "Maximum distance between any two tumor points.",
        'Perimeter (cm)': "Perimeter length of the tumor boundary.",
        'Compactness': "Shape roundness; closer to 1 means more circular.",
        'Irregularity': "How complex or jagged the tumor boundary is; higher = more irregular.",

        # Intensity features
        'Mean Intensity': "Average MRI intensity within tumor region.",
        'Std Intensity': "Variation of pixel intensities (heterogeneity).",
        'Min Intensity': "Lowest intensity value in tumor region.",
        'Max Intensity': "Highest intensity value in tumor region.",

        # Texture features
        'Contrast': "Local intensity variation; high contrast indicates more texture complexity.",
        'Homogeneity': "Uniformity of gray-level distribution; higher = smoother texture.",
        'Entropy': "Randomness or complexity of intensity values within tumor."
    }
    return feature_explanations