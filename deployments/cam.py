import streamlit as st
import numpy as np
import cv2
import torch
import requests
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F

st.set_page_config(page_title="CAM Visualizer", layout="centered", page_icon="📊")

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def load_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()
    model.layer4.register_forward_hook(get_activation('final_conv'))
    return model

@st.cache_data
def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        return requests.get(url).text.split('\n')
    except:
        return str('Failed to fetch the labels')

model = load_model()
labels = load_labels()

def get_cam(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    return np.uint8(255 * cam)

def create_visualizations(original_img_np, cam_raw):
    h, w, _ = original_img_np.shape
    
    cam_resized = cv2.resize(cam_raw, (w, h))
    
    heatmap_bgr = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(original_img_np, 0.5, heatmap_rgb, 0.5, 0)
    
    return heatmap_rgb, overlay

st.title("Class Activation Mapping (CAM) using ResNet18")
st.write("Upload an image to see what the CNN looks at when making a prediction.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file is not None:
    with st.spinner('Analyzing image...'):
        
        image_pil = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image_pil) 
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        input_tensor = preprocess(image_pil).unsqueeze(0)

        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[0][class_idx].item() * 100
        predicted_label = labels[class_idx]

        features = activation['final_conv'].detach().cpu().numpy()
        weight_softmax = model.fc.weight.detach().cpu().numpy()
        
        cam_raw = get_cam(features, weight_softmax, class_idx)
        
        viz_image_np = np.array(image_pil.resize((224, 224)))
        
        heatmap, overlay = create_visualizations(viz_image_np, cam_raw)

    st.success(f"Prediction: **{predicted_label}** ({confidence:.2f}%)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(viz_image_np, caption="1. Original Image", width='content')
    with col2:
        st.image(heatmap, caption="2. Heatmap", width='content')
    with col3:
        st.image(overlay, caption="3. Overlay", width='content')

    st.markdown("---")
    st.info("**Explanation:** \n- **Red areas** indicate where the model found strong evidence for the class.\n- **Blue areas** indicate low relevance.")
