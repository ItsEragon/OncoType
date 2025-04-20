# 🧬 OncoType

OncoType is a deep learning-powered system for detecting and classifying blood cancer cells from peripheral blood smear images. Using convolutional neural networks and Grad-CAM visualizations, OncoType assists in identifying malignant patterns and supports faster, more accurate diagnostics.


## 🖼️ Sample Grad-CAM Outputs

![Sample Image](/sample_grad_cams/gradcam_out%20(1).jpg) ![Sample Image](/sample_grad_cams/gradcam_out%20(2).jpg)


## 🚀 Features
Our technology can be applied across various medical and research domains, including:

1. Early detection and diagnosis of blood cancers such as leukemia

2. Assisting pathologists in analyzing peripheral blood smear images

3. Supporting medical education and hematology training

4. Enhancing research in automated disease detection and digital pathology

## 🧪 Usage
There are two main ways to use this model:

Option 1: Try the model online
You can run the model directly in your browser via Hugging Face Spaces:
   [OncoType](https://huggingface.co/spaces/ItsErAgOn/OncoType)


Option 2: Run Locally
Clone this repository:

   ```bash
   git clone https://github.com/ItsEragon/OncoType.git

Install required dependencies:

   pip install -r requirements.txt

Open the training and inference notebook:

Launch training.ipynb using Jupyter or VS Code.

Follow the final inference cells to load the model and test images.

Grad-CAM outputs and predictions will be shown in the notebook.
