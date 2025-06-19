# 🔍 Fingerprint Classification – ML Model - Partial Fingerprint Recognizer 

Welcome to the public portion of project **Half Mark**. This repository focuses specifically on the **model training** and **inference (prediction)** stages of the system, using deep learning techniques for fingerprint classification and partial fingerprint recognition.

> 🛡️ **Note**: The complete project includes a full-stack interface, extended preprocessing pipeline, and user interaction modules which are **private** and can be shared **upon request**.

---

## 📌 Project Summary

The core objective of this project is to classify fingerprint images into one of the following major fingerprint pattern types:

- **left_loop**
- **whirl**
- **right_loop**
- **tented_arch**
- **arch**

To accomplish this, we utilized **Transfer Learning** with the **ResNet50** architecture, leveraging its power to extract deep spatial features from fingerprint images.

---

## 🧠 Model Details

- 📚 **Architecture**: ResNet50 (pre-trained on ImageNet, fine-tuned on fingerprint data)
- 🛠️ **Framework**: TensorFlow / Keras
- 🗂️ **Classes**: left_loop, whirl, right_loop, tented_arch, arch
- 🧪 **Training**: NIST fingerprint dataset with data preprocessing techniques
- 💾 **Output**: `.h5` and `.pkl` files containing the trained model — **ready for deployment**

---

## 📂 Dataset Description

Dataset used for testing and training purposes: **NIST Special Database 4**, 8-Bit Gray Scale Images of Fingerprint Image Groups.  
[Link to the dataset provided in the description].  

### Dataset Details:
- Contains **4000 images** labeled as left_loop, whirl, right_loop, tented_arch, arch.
- Split for training and validation:
  - **3200 images** used for training  
  - **800 images** used for validation  

### Training Setup:
- A **50-layer Residual Neural Network (ResNet50)** was used to train the deep learning model.
- The final layer includes a **softmax activation unit** to estimate the probabilities of the input image belonging to specific classes.

### Training Results:
- **Training Accuracy**: ~91.1%  
- **Validation Accuracy**: ~83-86%

---

## 📁 What’s Included in This Repo

- ✅ Preprocessing logic for input images
- ✅ ResNet50-based model definition and training code
- ✅ Script for running predictions on new fingerprint images
- ✅ The **trained `.h5` and `.pkl` model files** for instant use — no need to retrain unless desired

---

## ❌ What’s Not Included (But Available on Demand)

The following components are **not published** on this public repo, but are available upon request:

- Full **frontend UI**
- **Desktop app interface**
- Additional documentation, deployment scripts, and backend integration
- Access to the **entire dataset**

---

## 📥 Access the Model and Dataset

Feel free to reach out to me for access to:  
1. The trained model files (`.h5` and `.pkl`).  
2. The complete dataset used for training and validation.  
3. Full **frontend UI**.  
4. **Desktop app interface**.  
7. Additional **documentation**, deployment scripts, and backend integration.  

**Contact me via LinkedIn or Email** for further information.

---

## 🖼️ Home Screen (UI Preview)

A screenshot of the home screen from the private part of the project is included below to give a visual sense of the complete system:

![Home Screen Preview](./assets/Home_Page.png)  

---

For questions or further assistance, don’t hesitate to reach out. Let’s build together!
