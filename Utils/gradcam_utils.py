import numpy as np
import torch
import cv2
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_multi_gradcam(model, input_tensor, target_category=None,
                           method='gradcam', target_layer_name='layer4[-2]',
                           smooth=False):
    """
    Generate Grad-CAM heatmap using different versions and layers.
    
    Args:
        model: torch model (already in eval mode)
        input_tensor: torch tensor [1, C, H, W]
        target_category: int, class index
        method: 'gradcam', 'gradcam++', 'scorecam'
        target_layer_name: str, e.g., 'layer4[-2]', 'layer3[-1]'
        smooth: bool, apply Gaussian blur
    
    Returns:
        grayscale_cam: np.array [H, W], normalized 0-1
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    # Map string to actual layer
    def get_layer(model, name):
        # support simple forms like 'layer4[-2]'
        layer_parts = name.split('[')
        base = getattr(model, layer_parts[0])
        if len(layer_parts) > 1:
            idx = int(layer_parts[1].rstrip(']'))
            return base[idx]
        return base
    
    target_layer = get_layer(model, target_layer_name)
    target_layers = [target_layer]

    # Choose method
    if method.lower() == 'gradcam':
        cam_method = GradCAM
    elif method.lower() == 'gradcam++':
        cam_method = GradCAMPlusPlus
    elif method.lower() == 'scorecam':
        cam_method = ScoreCAM
    else:
        raise ValueError("method must be 'gradcam', 'gradcam++', or 'scorecam'")
    
    cam = cam_method(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # Apply smoothing if needed
    if smooth:
        grayscale_cam = cv2.GaussianBlur(grayscale_cam, (5, 5), 0)

    # Normalize 0-1
    grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
    
    return grayscale_cam
