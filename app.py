import streamlit as st
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CONFIGURATION ==========
MODEL_PATH = 'best_model_m1_notebook.pt'
IMG_SIZE = 224
NUM_CLASSES = 2
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
OPTIMAL_THRESHOLD = 0.70

# ========== LOAD MODEL (CACHED) ==========
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

# ========== IMAGE PREPROCESSING ==========
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========== MAKE PREDICTION ==========
def predict(model, image):
    """
    Takes a PIL image and returns prediction probabilities and class label
    """
    img_tensor = val_transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
    
    prob_yes = probs[0]
    prob_no = probs[1]
    
    prediction = "TUMOR DETECTED" if prob_yes >= OPTIMAL_THRESHOLD else "NO TUMOR"
    confidence = max(prob_yes, prob_no)
    
    return {
        'prediction': prediction,
        'prob_yes': prob_yes,
        'prob_no': prob_no,
        'confidence': confidence,
        'threshold': OPTIMAL_THRESHOLD
    }

# ========== MAIN APP ==========
def main():
    st.markdown("<h1 style='text-align:center; color:#0066CC'>🧠 Brain Tumor Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#666666'>Upload a brain scan image to get predictions from the AI model</p>", unsafe_allow_html=True)

    # Add custom CSS
    st.markdown("""
        <style>
        .stFileUploader > div > div > input,
        .stTextInput > div > div > input {
            box-shadow: inset 2px 2px 5px rgba(0,0,0,0.2), 
                        inset -2px -2px 5px rgba(255,255,255,0.7);
            border: 1px solid #ccc;
            border-radius: 8px;
            background: linear-gradient(145deg, #f0f0f0, #ffffff);
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar info
    with st.sidebar:
        st.header("Model Information")
        st.metric("Model", "ResNet18")
        st.metric("Test F1-Score", "0.9651")
        st.metric("Independent Dataset F1", "0.9926")
        st.metric("ROC AUC", "0.9999")
        st.metric("Sensitivity (TPR)", "0.9883")
        st.metric("Specificity (TNR)", "0.9975")
        st.metric("Decision Threshold", f"{OPTIMAL_THRESHOLD}")
        st.divider()
        st.write("**Model Performance:**")
        st.write("- Catches 98.83% of tumors")
        st.write("- False alarm rate: 0.25%")
        st.write("- Trained on 1,356 images")

    # Load model once
    model = load_model()

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a brain scan image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a grayscale or color brain scan image"
        )

    with col2:
        st.subheader("Prediction Result")
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file).convert('RGB')
            
            # Make prediction
            result = predict(model, image)
            
            # Display prediction with color coding
            if result['prediction'] == "TUMOR DETECTED":
                st.error(f"### {result['prediction']}")
                color = "#FF4444"
            else:
                st.success(f"### {result['prediction']}")
                color = "#44FF44"
            
            # Display confidence
            st.metric(
                "Model Confidence",
                f"{result['confidence']:.2%}",
                help="How confident is the model in this prediction"
            )

    # Display image and detailed results
    if uploaded_file is not None:
        col_img, col_details = st.columns([1, 1])
        
        with col_img:
            st.subheader("Uploaded Image")
            st.image(image, use_column_width=True)
        
        with col_details:
            st.subheader("Prediction Details")
            
            st.write("**Probability Scores:**")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Tumor (Yes)", f"{result['prob_yes']:.4f}")
            with col_p2:
                st.metric("No Tumor", f"{result['prob_no']:.4f}")
            
            st.write("**Classification Logic:**")
            st.write(f"- If Tumor probability >= {OPTIMAL_THRESHOLD} --> Predict TUMOR")
            st.write(f"- If Tumor probability < {OPTIMAL_THRESHOLD} --> Predict NO TUMOR")
            
            st.divider()
            
            # Probability bar chart
            fig, ax = plt.subplots(figsize=(8, 3))
            categories = ['Tumor', 'No Tumor']
            probabilities = [result['prob_yes'], result['prob_no']]
            colors = ['#FF6B6B', '#4ECDC4']
            
            bars = ax.barh(categories, probabilities, color=colors)
            ax.axvline(OPTIMAL_THRESHOLD, color='black', linestyle='--', linewidth=2, label=f'Threshold: {OPTIMAL_THRESHOLD}')
            ax.set_xlim([0, 1])
            ax.set_xlabel('Probability', fontsize=11, fontweight='bold')
            ax.set_title('Model Output Probabilities', fontsize=12, fontweight='bold')
            ax.legend()
            
            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                ax.text(prob + 0.02, i, f'{prob:.4f}', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    # Information section
    st.divider()
    st.subheader("About This Model")

    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        st.write("**Training Data**")
        st.write("- 1,356 brain scans")
        st.write("- 856 with tumors (63%)")
        st.write("- 500 without tumors (37%)")

    with col_info2:
        st.write("**Architecture**")
        st.write("- ResNet18 (transfer learning)")
        st.write("- Fine-tuned on brain scans")
        st.write("- Threshold: 0.70")

    with col_info3:
        st.write("**Validation**")
        st.write("- Test F1: 0.9651")
        st.write("- Independent F1: 0.9926")
        st.write("- ROC AUC: 0.9999")

    st.divider()

    # Disclaimer
    st.warning(
        """
        DISCLAIMER: This model is for research and demonstration purposes only. 
        It should NOT be used for clinical diagnosis without validation by qualified medical professionals. 
        Always consult with a radiologist or medical doctor for actual diagnosis and treatment.
        """
    )

    # Add batch processing option
    st.divider()
    st.subheader("Batch Processing")

    uploaded_files = st.file_uploader(
        "Upload multiple images for batch predictions",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload multiple brain scan images at once",
        key="batch_uploader"
    )

    if uploaded_files and len(uploaded_files) > 0:
        st.write(f"Processing {len(uploaded_files)} images...")
        
        results_list = []
        progress_bar = st.progress(0)
        
        for idx, file in enumerate(uploaded_files):
            image = Image.open(file).convert('RGB')
            result = predict(model, image)
            results_list.append({
                'Filename': file.name,
                'Prediction': result['prediction'],
                'Tumor Prob': f"{result['prob_yes']:.4f}",
                'Confidence': f"{result['confidence']:.2%}"
            })
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # Display results table
        st.subheader("Batch Results")
        st.dataframe(results_list, use_container_width=True)
        
        # Summary statistics
        tumor_count = sum(1 for r in results_list if r['Prediction'] == 'TUMOR DETECTED')
        no_tumor_count = len(results_list) - tumor_count
        
        col_sum1, col_sum2 = st.columns(2)
        with col_sum1:
            st.metric("Tumors Detected", tumor_count)
        with col_sum2:
            st.metric("No Tumor", no_tumor_count)


main()