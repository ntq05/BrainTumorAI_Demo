import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def enable_dropout(model):
    """ Enable dropout layers during inference. """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def predict_with_mc_dropout_classification(model, img_tensor, n_iter=10):
    """
    img_tensor: (1, C, H, W) preprocessed
    returns: mean_probs (1, num_classes), std_probs (1, num_classes)
    """
    model.eval()
    enable_dropout(model)  # keep dropout active
    probs = []
    with torch.no_grad():
        for _ in range(n_iter):
            out = model(img_tensor)                     # logits
            p = F.softmax(out, dim=1).cpu().numpy()     # (1, num_classes)
            probs.append(p)
    probs = np.concatenate(probs, axis=0)  # (n_iter, 1, num_classes)
    mean_probs = probs.mean(axis=0)        # (1, num_classes)
    std_probs = probs.std(axis=0)          # (1, num_classes)
    return mean_probs, std_probs

def predict_image_mc(image_path, model, class_names, n_iter=10):
    model.to(device)
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform_test(img).unsqueeze(0).to(device)

    mean_probs, std_probs = predict_with_mc_dropout_classification(model, img_tensor, n_iter=n_iter)

    pred_class = int(np.argmax(mean_probs))
    conf = float(mean_probs[pred_class])
    conf_std = float(std_probs[pred_class])

    return pred_class, mean_probs, std_probs