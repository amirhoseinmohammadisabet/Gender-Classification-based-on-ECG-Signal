# ECG Signal Analysis for Sex Classification  

## 📌 Overview  
This project explores the use of **electrocardiogram (ECG) signals** to differentiate between male and female individuals. Leveraging the **Autonomic Aging dataset**, we preprocess ECG data, extract features (FFT, time/frequency-domain), and evaluate machine learning models for classification.  

**Key Result**:  
- **Best Model**: Decision Tree (**89.50% accuracy**).  
- **Robust Alternatives**: ANN & Naive Bayes (~79% accuracy).  

---

## 🔍 Methodology  
### 1. **Preprocessing**  
- Noise removal (baseline wander, powerline interference, high-frequency artifacts).  
- Butterworth filters, wavelet denoising, and notch filtering.  

### 2. **Feature Extraction**  
- **Fiducial**: QRS complex, P/T waves.  
- **Non-Fiducial**: FFT-based (spectral entropy, dominant frequency).  
- **Dimensionality Reduction**: PCA.  

### 3. **Models Evaluated**  
| Model               | Accuracy | Cross-Val Accuracy |  
|---------------------|----------|---------------------|  
| Decision Tree       | 89.50%   | 60.37%             |  
| Random Forest       | 63.20%   | 62.47%             |  
| SVM                 | 57.90%   | 61.14%             |  
| Naive Bayes         | 78.90%   | 61.28%             |  
| ANN                 | 78.90%   | **63.62%**         |  

---

## 🚀 How to Run  
1. **Clone the repo**:  
   ```bash  
   git clone https://github.com/amirhoseinmohammadisabet/Health.git
   
