# Brain Tumor Detection Web App

An interactive **AI-powered web application** built with **Streamlit** and **PyTorch** that detects the presence of brain tumors from MRI scan images.  
The model is based on **ResNet18** and fine-tuned on 1,356 labeled brain scan images.

---

## Features

- Upload a **single MRI image** or **multiple images** for batch prediction  
- Displays **probability scores** and **model confidence**  
- Visualizes classification threshold and output probabilities  
- Provides model performance metrics (F1-score, ROC AUC, sensitivity, specificity)  
- Interactive, no coding required — just upload an image!  
- Includes disclaimer for non-clinical use  

---

## Tech Stack

| Component | Description |
|------------|-------------|
| **Frontend** | Streamlit |
| **Backend** | PyTorch |
| **Model** | ResNet18 (transfer learning) |
| **Visualization** | Matplotlib |
| **Metrics** | scikit-learn (ROC, AUC) |

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/brain-tumor-detection.git
cd brain-tumor-detection
