import sys
import os
from dotenv import load_dotenv
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

st.set_page_config(page_title="Brain Tumor AI Assistant", layout="wide")


import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from Utils.gradcam_utils import generate_multi_gradcam
from Utils.MRI_features import compute_tumor_features, features_note
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.models as models
import segmentation_models_pytorch as smp
from Utils.Report_Generator import generate_tumor_report
import torch.nn as nn
from Utils.MC_Dropout_Classification import predict_image_mc



# ---------- Load environment ----------
load_dotenv()
URL = os.getenv("URL")
MODEL = os.getenv("MODEL")

# ---------- Load Models ----------
@st.cache_resource
def load_models():
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = resnet18.fc.in_features
    resnet18.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, 4)
    )

    # Load state_dict
    resnet_state_dict = torch.load('Models/Classification_Models/best_resnet18.pth', map_location='cpu')
    resnet18.load_state_dict(resnet_state_dict)
    resnet18.eval()


    resunet = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    resunet_state_dict = torch.load('Models/Segmentation_Models/best_resunet.pth', map_location='cpu')
    resunet.load_state_dict(resunet_state_dict)
    resunet.eval()

    return resnet18, resunet

resnet18, resunet = load_models()

# ---------- Preprocessing ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------- Streamlit App ----------

st.title("🧠 Brain Tumor AI Assistant")
st.caption("Upload an MRI image to classify, localize, and segment brain tumors. This tool is AI-generated and **not a substitute for professional medical advice.**")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # ----- Classification -----
    img_tensor = transform(image).unsqueeze(0)
    labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

    pred_class, mean_probs, std_probs = predict_image_mc(
        image_path=uploaded_file,     # nếu uploaded_file là đường dẫn
        model=resnet18,
        class_names=labels,
        n_iter=10                     # số lần Monte Carlo sampling
    )
    predicted_label = labels[pred_class]

    labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    labels_vi = [
        "U thần kinh đệm", 
        "U màng não", 
        "Không có khối u", 
        "U tuyến yên"
    ]

    st.markdown(f"### 🧩 Predicted Tumor Type: **{predicted_label} ({labels_vi[pred_class]})**")
    st.markdown(
        f"<p style='font-size:20px; color:#1E90FF; font-weight:bold;'>Confidence: {mean_probs[pred_class]:.4f} ± {std_probs[pred_class]:.4f}</p>",
        unsafe_allow_html=True
    )

    # 3-column layout for images
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption='Original MRI', use_column_width=True)

    if predicted_label != 'No Tumor':
        # ----- Grad-CAM -----
        grayscale_cam = generate_multi_gradcam(
            resnet18, img_tensor, target_category=pred_class,
            method='gradcam++', target_layer_name='layer4[-2]', smooth=True
        )
        rgb_img = np.array(image.resize((224, 224))) / 255.0
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        with col2:
            st.image(cam_image, caption='Grad-CAM (Model Attention)', use_column_width=True)

        # ----- Segmentation -----
        img_resized = image.resize((256, 256))
        img_tensor_seg = transforms.ToTensor()(img_resized).unsqueeze(0)
        with torch.no_grad():
            mask_pred = torch.sigmoid(resunet(img_tensor_seg)).squeeze().numpy()
        mask_pred = (mask_pred > 0.5).astype(np.uint8) * 255

        mask_color = cv2.applyColorMap(mask_pred, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.array(img_resized), 0.6, mask_color, 0.4, 0)
        with col3:
            st.image(overlay, caption='Segmentation (ResUNet)', use_column_width=True)

        # ----- Tumor Features -----
        st.markdown("### 📊 Tumor Quantitative Features")
        img_gray = np.array(img_resized.convert('L'))
        tumor_features = compute_tumor_features(mask_pred, img_gray, voxel_size=(0.1, 0.1))

        feature_expander = st.expander("Show extracted features", expanded=False)
        with feature_expander:
            for k, v in tumor_features.items():
                explanation = features_note().get(k, "")
                if isinstance(v, (list, np.ndarray)):
                    st.write(f"**{k}**: {v} — {explanation}")
                else:
                    st.write(f"**{k}**: {v:.2f} — {explanation}")

        # ----- Report -----
        st.markdown("### 🧾 AI Tumor Report")
        report = generate_tumor_report(
            tumor_features,
            URL=URL,
            model=MODEL,
            predicted_label=predicted_label,
            lang='en'
        )

    else:
        with col2:
            st.markdown("### No Tumor Detected")
            st.info("The model did not detect any tumor in the uploaded MRI image.")
