import streamlit as st
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Model setup
NUM_CLASSES = 4  # Updated to match the trained model
CLASS_NAMES = ['Class1', 'Class2', 'Class3', 'Class4']  # Updated to match the trained model
SEVERITY_MSG = {
    0: "A benign hematological entity exhibiting no signs of malignant transformation. The cellular morphology remains consistent with normal hematopoiesis, with preserved nuclear-to-cytoplasmic ratios, orderly chromatin, and the absence of atypical mitotic figures. No immediate clinical concern; routine surveillance may suffice.",
    1: "An early-stage precursor B-cell malignancy, marked by subtle yet significant deviations from normal lymphopoiesis. These cells begin to demonstrate nuclear irregularities, increased nuclear-cytoplasmic ratio, and early chromatin dispersion — a harbinger of uncontrolled proliferation if left unchecked. Clinical intervention is crucial at this incipient stage.",
    2: "A progressed pre-B lymphoblast population displaying clear morphological evidence of malignancy. Nuclear convolutions, prominent nucleoli, and cytoplasmic basophilia are characteristic. The disease at this stage possesses high proliferative potential, posing a significant systemic threat. Prompt and aggressive therapy is often indicated to curtail disease progression.",
    3: "A highly aggressive and immature leukemic state, where pro-B lymphoblasts dominate. These cells exhibit profound anaplasia, scant cytoplasm, and dense chromatin irregularities. Rapid clinical deterioration is a hallmark; immediate, intensive therapeutic strategies are imperative for any hope of remission."
}  # Updated to match the trained model

def build_model():
    m = models.resnet50(pretrained=True)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m

# Load the trained model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model().to(DEVICE)
model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
model.eval()

# Image transformations
from torchvision import transforms
tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Page configuration
st.set_page_config(
    page_title="OncoType",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS with navbar styling
st.markdown("""
<style>
    /* Basic styling that's more compatible with Streamlit */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    h1, h2, h3, h4 {
        font-family: 'Poppins', sans-serif;
        color: #8bb9fe;
    }
    
    .stApp {
        background-color: #0e1117;
        background-image: url('https://images.unsplash.com/photo-1534972195531-d756b9bfa9f2?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        background-blend-mode: overlay;
    }
    
    /* Top header - improved styling */
    .top-header {
        background-color: rgba(14, 17, 23, 0.95);
        color: #ffffff;
        text-align: center;
        padding: 0.5rem;
        font-size: 0.85rem;
        border-bottom: 1px solid rgba(139, 185, 254, 0.15);
    }
    
    /* Navbar - completely redesigned */
    .navbar {
        position: sticky;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999;
        background-color: rgba(14, 17, 23, 0.97);
        backdrop-filter: blur(15px);
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 3rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        border-bottom: 1px solid rgba(139, 185, 254, 0.1);
        border-radius: 15px;
    }
    
    .navbar a {
        text-decoration: none !important;
    }
    
    .navbar-brand {
        display: flex;
        align-items: center;
        color: #8bb9fe;
        font-weight: 700;
        font-size: 1.6rem;
        text-decoration: none !important;
        letter-spacing: 0.5px;
    }
    
    .navbar-brand img {
        height: 32px;
        width: 32px;
        margin-right: 12px;
        display: inline-block;
        vertical-align: middle;
    }
    
    .navbar-links {
        display: flex;
        gap: 1.5rem;
        align-items: center;
    }
    
    .navbar-link {
        color: #e1e1e1;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s;
        padding: 0.6rem 1.2rem;
        border-radius: 30px;
        font-size: 0.95rem;
    }
    
    .navbar-link:hover {
        color: #ffffff;
        background-color: rgba(139, 185, 254, 0.15);
        transform: translateY(-1px);
    }
    
    .navbar-link.active {
        background-color: rgba(139, 185, 254, 0.2);
        color: #8bb9fe;
    }
    
    .github-button {
        background: linear-gradient(135deg, #4c6ef5, #3b5bdb);
        color: #ffffff !important;
        padding: 0.6rem 1.2rem;
        border-radius: 30px;
        text-decoration: none;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
        transition: all 0.3s;
        box-shadow: 0 4px 10px rgba(59, 91, 219, 0.3);
    }
    
    .github-button:hover {
        background: linear-gradient(135deg, #5c7cfa, #4263eb);
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(59, 91, 219, 0.4);
        color: #ffffff !important;
    }
    
    /* Content wrapper - improved spacing */
    .content-wrapper {
        max-width: 1400px;
        margin: 0 auto;
        padding: 1.5rem;
    }
    
    /* Hero section styling */
    .hero-section {
        margin-top: 1rem;
        margin-bottom: 3rem;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, rgba(25, 25, 50, 0.6), rgba(14, 17, 23, 0.85));
        border-radius: 15px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(139, 185, 254, 0.1);
        backdrop-filter: blur(10px);
        text-align: center;
    }
    
    /* Main header - improved styling */
    .main-header {
        text-align: center;
        margin-bottom: 2.5rem;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(25, 25, 50, 0.6), rgba(14, 17, 23, 0.7));
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        margin-bottom: 1rem;
        color: #8bb9fe;
        font-weight: 700;
        letter-spacing: 1px;
        text-shadow: 0 2px 10px rgba(139, 185, 254, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .main-header h1 img {
        height: 48px;
        width: 48px;
        margin-right: 15px;
        display: inline-block;
        vertical-align: middle;
    }
    
    .main-header p {
        color: #e1e1e1;
        font-size: 1.3rem;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.7;
    }
    
    /* Feature cards - improved styling */
    .feature-container {
        display: flex;
        justify-content: space-around;
        margin: 2.5rem 0;
        flex-wrap: wrap;
        gap: 1.5rem;
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(30, 30, 60, 0.4), rgba(14, 17, 23, 0.7));
        padding: 2rem;
        border-radius: 15px;
        width: 100%;
        max-width: 300px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(139, 185, 254, 0.08);
        transition: all 0.3s;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(139, 185, 254, 0.15);
    }
    
    .feature-card i {
        font-size: 2.5rem;
        color: #8bb9fe;
        margin-bottom: 1.5rem;
        display: block;
    }
    
    .feature-card h3 {
        margin-bottom: 1rem;
        font-size: 1.4rem;
    }
    
    .feature-card p {
        color: #d1d1d1;
        line-height: 1.6;
    }
    
    /* Section styling - enhanced */
    .section {
        margin-bottom: 4rem;
        padding: 2.5rem;
        background: linear-gradient(135deg, rgba(25, 25, 50, 0.5), rgba(14, 17, 23, 0.7));
        border-radius: 15px;
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(139, 185, 254, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .section h2 {
        color: #8bb9fe;
        margin-bottom: 1.5rem;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        position: relative;
        display: inline-block;
    }
    
    .section h2:after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 0;
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #8bb9fe, transparent);
    }
    
    .section p {
        color: #e1e1e1;
        font-size: 1.1rem;
        line-height: 1.7;
        margin-bottom: 1.5rem;
    }
    
    /* Caption result styles */
    .caption-result {
        background-color: rgba(30, 30, 60, 0.4);
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        border-left: 4px solid #8bb9fe;
    }
    
    .caption-text {
        font-size: 1.4rem;
        color: #f1f1f1;
        font-style: italic;
        line-height: 1.6;
    }
    
    /* Team member cards */
    .team-container {
        display: flex;
        gap: 2rem;
        flex-wrap: wrap;
        justify-content: center;
        margin: 2rem 0;
    }
    
    .team-member {
        background-color: rgba(30, 30, 60, 0.4);
        border-radius: 10px;
        padding: 1.5rem;
        width: 250px;
        text-align: center;
    }
    
    .team-member img {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        margin-bottom: 1rem;
    }
    
    .team-member h4 {
        margin-bottom: 0.5rem;
    }
    
    .team-member p {
        color: #a1a1a1;
        font-size: 0.9rem;
    }
    
    /* Contact section */
    .contact-info {
        background-color: rgba(30, 30, 60, 0.4);
        padding: 2rem;
        border-radius: 10px;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #a1a1a1;
        margin-top: 2rem;
        font-size: 0.9rem;
        background-color: rgba(14, 17, 23, 0.95);
        border-top: 1px solid rgba(139, 185, 254, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #4c6ef5 !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
    }
    
    .stButton > button:hover {
        background-color: #364fc7 !important;
    }
    
    /* Upload area - enhanced styling */
    .upload-box {
        background: linear-gradient(135deg, rgba(30, 30, 60, 0.6), rgba(14, 17, 23, 0.8));
        padding: 3rem 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2.5rem;
        border: 1px solid rgba(139, 185, 254, 0.1);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
        transition: all 0.3s;
    }
    
    .upload-box:hover {
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(139, 185, 254, 0.2);
    }
    
    /* For mobile devices */
    @media screen and (max-width: 768px) {
        .navbar {
            padding: 1rem;
            flex-direction: column;
            gap: 1rem;
            top: 60px; /* Adjusted for mobile */
        }
        
        .navbar-links {
            gap: 0.5rem;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .content-wrapper {
            margin-top: 150px; /* Adjusted for mobile */
        }
    }

    /* Center alignment container */
    .center-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        width: 100%;
        margin: 0 auto;
    }

    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        background: linear-gradient(135deg, rgba(30, 30, 60, 0.6), rgba(14, 17, 23, 0.8));
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(139, 185, 254, 0.1);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        margin: 0 auto 2rem auto;
        max-width: 450px;
        width: 100%;
    }

    .image-container img {
        border-radius: 8px;
        width: auto !important;
        height: auto;
        max-width: 400px !important;
        max-height: 400px;
        object-fit: contain;
        display: block;
        margin: 0 auto;
    }

    .status-container {
        background: rgba(15, 74, 0, 0.8);
        color: #ffffff;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        max-width: 300px;
        margin-left: auto;
        margin-right: auto;
    }

    .progress-container {
        max-width: 300px;
        margin: 1rem auto;
    }

    .section-header {
        text-align: center;
        margin-bottom: 2rem;
    }

    .section-header h2 {
        display: inline-block;
        position: relative;
        color: #8bb9fe;
        margin-bottom: 1.5rem;
        font-size: 2.2rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# Create navbar - improved structure and styling
st.markdown("""
<div class="navbar">
    <a href="#" class="navbar-brand">
        <img src="https://images.vexels.com/media/users/3/145502/isolated/preview/9824e521b3d8c5e4f893b269cf6f9128-breast-cancer-ribbon.png" alt="OncoType Logo">
        <span>OncoType</span>
    </a>
    <div class="navbar-links">
        <a href="#home" class="navbar-link active">Home</a>
        <a href="#about" class="navbar-link">About</a>
        <a href="#contact" class="navbar-link">Contact</a>
        <a href="https://github.com/ItsEragon/SmartCaption" target="_blank" class="github-button">
            <svg height="16" width="16" viewBox="0 0 16 16" fill="white">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
            Documentation
        </a>
    </div>
</div>
<div class="content-wrapper">
""", unsafe_allow_html=True)

# Hero Section with background
st.markdown("""
<div id="home" class="hero-section">
    <div class="main-header">
        <h1>
            <img src="https://images.vexels.com/media/users/3/145502/isolated/preview/9824e521b3d8c5e4f893b269cf6f9128-breast-cancer-ribbon.png" alt="OncoType Logo">
            OncoType
        </h1>
        <p>Transform microscopic images of blood smears into detailed descriptions with our advanced AI vision technology. Powered by state-of-the-art machine learning models to see and understand the Acute Lymphoblastic Leukemia(ALL),  i.e. Blood Cancer.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Features section with enhanced styling
st.markdown("""
<div class="feature-container">
    <div class="feature-card">
        <i>🚀</i>
        <h3>Advanced AI</h3>
        <p>Our model is trained on thousands of diverse images for accurate and contextual captioning</p>
    </div>
    <div class="feature-card">
        <i>⚡</i>
        <h3>Fast Processing</h3>
        <p>Get detailed results in seconds with our optimized neural network pipeline and cloud infrastructure</p>
    </div>
    <div class="feature-card">
        <i>🔍</i>
        <h3>Detail Detection</h3>
        <p>Identifies various stages of Blood Cancer, stating their severity with remarkable precision</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Upload section with enhanced styling
st.markdown('<div class="upload-box"><h2>🩸 Upload Your Image</h2><p style="color: #d1d1d1; margin-bottom: 1.5rem;">Drop your image below and let AI reveal what it sees</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

# Process image and show results
if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Resize image to match model input size
    display_size = (400, 400)  # Increased from 250x250 to 400x400
    model_size = (224, 224)    # Size for model input
    
    # Create display version of the image
    display_image = image.resize(display_size, Image.Resampling.LANCZOS)
    
    # Input Image Section
    st.markdown("""
        <div class="section-header" style="text-align: center;">
            <h2 style="display: inline-block;">📸 Your Image</h2>
        </div>
        <div class="center-container">
            <div class="image-container">
                <img src="data:image/png;base64,{}"/>
            </div>
        </div>
    """.format(image_to_base64(display_image)), unsafe_allow_html=True)
    
    # Model Analysis Section
    st.markdown("""
        <div class="section-header" style="text-align: center;">
            <h2 style="display: inline-block;">🔍 Model Analysis</h2>
        </div>
        <div class="center-container">
    """, unsafe_allow_html=True)
    
    # Progress container
    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate processing
    for i in range(101):
        progress_bar.progress(i)
        if i < 30:
            status_text.text("🧠 Analyzing image content...")
        elif i < 60:
            status_text.text("👁️ Identifying objects and context...")
        elif i < 90:
            status_text.text("✍️ Generating natural language description...")
        else:
            status_text.text("🔮 Finalizing analysis...")
        time.sleep(0.02)
    
    # Process image with model
    inp = tfms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(inp)
    cls = out.argmax(1).item()
    caption = SEVERITY_MSG[cls]
    
    # Generate Grad-CAM visualization
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    gcam = cam(input_tensor=inp, targets=[ClassifierOutputTarget(cls)])[0]
    rgb = np.array(image.resize(model_size), dtype=float) / 255
    vis = show_cam_on_image(rgb, gcam, use_rgb=True)
    vis_pil = Image.fromarray((vis * 255).astype(np.uint8))
    vis_display = vis_pil.resize(display_size, Image.Resampling.LANCZOS)  # Using same display_size as input image
    
    # Show success message
    st.markdown('<div class="status-container">✅ Analysis completed successfully!</div>', unsafe_allow_html=True)
    
    # Display Grad-CAM visualization
    st.markdown("""
        <div class="center-container">
            <div class="image-container">
                <img src="data:image/png;base64,{}"/>
            </div>
        </div>
    """.format(image_to_base64(vis_display)), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display results
    st.markdown("<h3 style='text-align: center;'>🔮 AI Analysis Results</h3>", unsafe_allow_html=True)
    st.markdown(f'<div class="caption-result"><p class="caption-text">"{caption}"</p></div>', unsafe_allow_html=True)
    
    # Single Save Results button with proper alignment
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.download_button(
            "💾 Save Results",
            f"Prediction: {caption}",
            "analysis.txt",
            use_container_width=True  # Makes the button use full column width
        )
        
else:
    # Placeholder with enhanced styling when no image is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 3.5rem 2rem; background: linear-gradient(135deg, rgba(30, 30, 60, 0.6), rgba(14, 17, 23, 0.8)); border-radius: 15px; border: 1px solid rgba(139, 185, 254, 0.1); box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);">
        <img src="https://img.icons8.com/fluency/96/000000/image.png" alt="Upload illustration" style="width: 96px; height: 96px; margin-bottom: 1.5rem;">
        <h3 style="color: #8bb9fe; margin-bottom: 1rem; font-size: 1.8rem;">Ready to see the magic?</h3>
        <p style="color: #e1e1e1; font-size: 1.1rem; margin-bottom: 1.5rem;">Upload an image to get an AI-generated description</p>
        <p style="color: #c1c1c1; font-size: 0.9rem;">Supports JPG, JPEG, and PNG formats</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</section>', unsafe_allow_html=True)

# --------------------- ABOUT SECTION ---------------------

st.markdown("""
<div id="about" style="padding-top: 60px; margin-top: -60px;">
<h2 style="text-align: center; color: #8bb9fe;">About OncoType</h2>
""", unsafe_allow_html=True)

st.markdown("""
<p>OncoType is a cutting-edge AI application that leverages state-of-the-art deep learning models to generate accurate and natural language descriptions of microscopic images of blood smears. Our mission is to make visual content more accessible and understandable for everyone.</p>

<p>Our technology can be applied across various medical and research domains, including:</p>
<ul style="color: #e1e1e1; margin-bottom: 2rem; padding-left: 1.5rem;">
    <li style="margin-bottom: 0.5rem;">Early detection and diagnosis of blood cancers such as leukemia</li>
    <li style="margin-bottom: 0.5rem;">Assisting pathologists in analyzing peripheral blood smear images</li>
    <li style="margin-bottom: 0.5rem;">Supporting medical education and hematology training</li>
    <li style="margin-bottom: 0.5rem;">Enhancing research in automated disease detection and digital pathology</li>
</ul>

<h2 style="color: #8bb9fe; margin: 2rem 0 1.5rem; font-size: 1.8rem; text-align: center;">Meet Our Team</h2>
""", unsafe_allow_html=True)

# Team members with enhanced styling
st.markdown("""
<div class="team-container">
    <div class="team-member">
        <img src="https://pbs.twimg.com/profile_images/1758212195349835776/1PL3ch0G_400x400.jpg" alt="Team Member">
        <h4>Reek Das</h4>
        <p style="color: #8bb9fe; margin-bottom: 0.7rem;">ML Engineer</p>
        <p>A curious student diving deep into machine learning, with a sharp focus on making machines see and understand the world through computer vision.</p>
    </div>
    <div class="team-member">
        <img src="https://pbs.twimg.com/profile_images/1102831027423068162/yPb4Qe06_400x400.jpg" alt="Team Member">
        <h4>Amritpal Singh</h4>
        <p style="color: #8bb9fe; margin-bottom: 0.7rem;">Developer</p>
        <p>A passionate student developer crafting seamless user experiences, specializing in turning ideas into interactive frontends.</p>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</section>', unsafe_allow_html=True)

# --------------------- CONTACT SECTION ---------------------

st.markdown("""
<div id="contact" style="padding-top: 60px; margin-top: -60px;">
<h2 style="text-align: center; color: #8bb9fe;">Contact Us</h2>
""", unsafe_allow_html=True)

# Social Media Section with enhanced styling
st.markdown("""
<div style="display: flex; justify-content: center; margin: 2rem auto; max-width: 500px;">
    <!-- Social Media Section -->
    <div style="width: 100%; background: linear-gradient(135deg, rgba(30, 30, 60, 0.6), rgba(14, 17, 23, 0.8)); padding: 2.5rem; border-radius: 15px; border: 1px solid rgba(139, 185, 254, 0.1); box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);">
        <p style="color: #e1e1e1; margin-bottom: 2rem; text-align: center;">Stay updated with our latest developments and join our community!</p>
        <div style="display: flex; flex-direction: column; gap: 1rem;">
            <a href="https://x.com/reek312" target="_blank" style="display: flex; align-items: center; gap: 1rem; background: rgba(14, 17, 23, 0.5); padding: 1rem; border-radius: 10px; text-decoration: none; color: #e1e1e1; transition: all 0.3s; hover: transform: translateY(-2px);">
                <span style="font-size: 1.5rem; min-width: 32px; text-align: center;">𝕏</span>
                <span>@reek312</span>
            </a>
            <a href="http://linkedin.com/in/itseragon" target="_blank" style="display: flex; align-items: center; gap: 1rem; background: rgba(14, 17, 23, 0.5); padding: 1rem; border-radius: 10px; text-decoration: none; color: #e1e1e1; transition: all 0.3s;">
                <span style="font-size: 1.5rem; min-width: 32px; text-align: center;">🔗</span>
                <span>Amritpal Singh</span>
            </a>
            <a href="https://github.com/ItsEragon" target="_blank" style="display: flex; align-items: center; gap: 1rem; background: rgba(14, 17, 23, 0.5); padding: 1rem; border-radius: 10px; text-decoration: none; color: #e1e1e1; transition: all 0.3s;">
                <span style="font-size: 1.5rem; min-width: 32px; text-align: center;">📦</span>
                <span>GitHub</span>
            </a>
            <a href="https://instagram.com/reek312" target="_blank" style="display: flex; align-items: center; gap: 1rem; background: rgba(14, 17, 23, 0.5); padding: 1rem; border-radius: 10px; text-decoration: none; color: #e1e1e1; transition: all 0.3s;">
                <span style="font-size: 1.5rem; min-width: 32px; text-align: center;">📸</span>
                <span>Instagram</span>
            </a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</section>', unsafe_allow_html=True)

# Close content wrapper
st.markdown('</div>', unsafe_allow_html=True)

# Add a subtle footer
st.markdown("""
<div style="text-align: center; padding: 1.5rem; margin-top: 2rem; color: #a1a1a1; font-size: 0.9rem; background: linear-gradient(180deg, rgba(14, 17, 23, 0), rgba(14, 17, 23, 0.95));">
    <p>OncoType © 2025 | Built with ❤️ and Streamlit</p>
    <p style="font-size: 0.8rem; color: #717171;">Images processed by this tool are not stored permanently and are only used for generating captions.</p>
</div>
""", unsafe_allow_html=True)

# Add custom JavaScript for smooth scrolling with offset
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Get all links that start with #
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            // Get the target element
            const targetId = this.getAttribute('href').slice(1);
            if (!targetId) return; // Handle empty hash
            
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                // Get the navbar height for offset
                const navbarHeight = document.querySelector('.navbar').offsetHeight;
                
                // Calculate the target position with offset
                const targetPosition = targetElement.getBoundingClientRect().top + window.pageYOffset - navbarHeight;
                
                // Smooth scroll to target
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
                
                // Update active state in navbar
                document.querySelectorAll('.navbar-link').forEach(link => {
                    link.classList.remove('active');
                });
                if (this.classList.contains('navbar-link')) {
                    this.classList.add('active');
                }
            }
        });
    });
});
</script>
""", unsafe_allow_html=True)

# Settings sidebar with enhanced styling
with st.sidebar:
    st.markdown('<h2 style="color: #8bb9fe; margin-bottom: 1.5rem; font-size: 1.8rem;">⚙️ Settings</h2>', unsafe_allow_html=True)
    
    st.markdown('<h3 style="color: #e1e1e1; margin: 1.5rem 0 1rem; font-size: 1.3rem; border-bottom: 1px solid rgba(139, 185, 254, 0.2); padding-bottom: 0.5rem;">Caption Options</h3>', unsafe_allow_html=True)
    caption_detail = st.slider("Detail Level", 1, 5, 3)
    include_objects = st.checkbox("Include objects", value=True)
    include_context = st.checkbox("Include scene context", value=True)
    include_colors = st.checkbox("Include colors", value=True)
    
    st.markdown('<h3 style="color: #e1e1e1; margin: 1.5rem 0 1rem; font-size: 1.3rem; border-bottom: 1px solid rgba(139, 185, 254, 0.2); padding-bottom: 0.5rem;">Model Options</h3>', unsafe_allow_html=True)
    model_type = st.selectbox("Model Type", ["Standard", "Detailed", "Creative"])
    language = st.selectbox("Language", ["English", "Spanish", "French", "German", "Japanese"])
    
    st.markdown('<h3 style="color: #e1e1e1; margin: 1.5rem 0 1rem; font-size: 1.3rem; border-bottom: 1px solid rgba(139, 185, 254, 0.2); padding-bottom: 0.5rem;">Quick Navigation</h3>', unsafe_allow_html=True)
    st.markdown('<div style="display: flex; flex-direction: column; gap: 0.7rem;">', unsafe_allow_html=True)
    st.markdown('<a href="#home" style="color: #8bb9fe; text-decoration: none; display: flex; align-items: center; padding: 0.5rem; background: rgba(139, 185, 254, 0.1); border-radius: 8px;"><span style="margin-right: 8px;">🏠</span> Home</a>', unsafe_allow_html=True)
    st.markdown('<a href="#about" style="color: #8bb9fe; text-decoration: none; display: flex; align-items: center; padding: 0.5rem; background: rgba(139, 185, 254, 0.1); border-radius: 8px;"><span style="margin-right: 8px;">ℹ️</span> About</a>', unsafe_allow_html=True)
    st.markdown('<a href="https://github.com/your-username/vision-oracle" style="color: #8bb9fe; text-decoration: none; display: flex; align-items: center; padding: 0.5rem; background: rgba(139, 185, 254, 0.1); border-radius: 8px;"><span style="margin-right: 8px;">📚</span> Documentation</a>', unsafe_allow_html=True)
    st.markdown('<a href="#contact" style="color: #8bb9fe; text-decoration: none; display: flex; align-items: center; padding: 0.5rem; background: rgba(139, 185, 254, 0.1); border-radius: 8px;"><span style="margin-right: 8px;">📧</span> Contact</a>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a subtle version info
    st.markdown("""
    <div style="margin-top: 3rem; padding-top: 1rem; border-top: 1px solid rgba(139, 185, 254, 0.1); text-align: center; font-size: 0.8rem; color: #717171;">
        <p>OncoType v1.0.2</p>
        <p>Last updated: June 2025</p>
    </div>
    """, unsafe_allow_html=True)
