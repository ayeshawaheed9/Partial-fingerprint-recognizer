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
- The dataset, provided by **NIST**, contains **4000 images** labeled into the following categories:  
  **left_loop, whirl, right_loop, tented_arch, arch**.
- Images were split into:
  - **3200 images** for training  
  - **800 images** for validation  

### Dataset Preparation:
To organize and preprocess the dataset for training:
- A **segmentation script** was developed to classify and separate images based on the metadata.
- The script automates the process of moving and categorizing fingerprint images into their respective class directories.  
- This script is a crucial part of the preprocessing pipeline and ensures the dataset is ready for model training.

### Access and Customization:
- The segmentation script and dataset preparation pipeline can be provided upon request.  
- If you'd like to split the dataset or customize the pipeline for your use case, the provided script and instructions will guide you through the process.
  
### Training Setup:
- After preprocessing, the dataset was used with additional data augmentation to enhance the training process.
- A **50-layer Residual Neural Network (ResNet50)** was used for training, with a **softmax activation unit** in the final layer to estimate the probabilities of the input image belonging to specific classes.

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
5. Additional **documentation**, deployment scripts, and backend integration.
6. **Classification Script** which is a crucial part of the preprocessing pipeline and ensures the dataset is ready for model training

---

## 🖼️ Home Screen (UI Preview)

A screenshot of the home screen from the private part of the project is included below to give a visual sense of the complete system:

![Home Screen Preview](./assets/Home_Page.png)  

---

## Creators
- **Syed Zair Hussain**
  - [GitHub](https://github.com/zairjafry)
  - [LinkedIn](https://www.linkedin.com/in/zairjafry)
- **Ayesha Waheed**
  - [Github](https://github.com/ayeshawaheed9)
  - [LinkedIn](https://www.linkedin.com/in/ayesha-waheed-?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3B1COxssSJS0qTJ8LneDEMNw%3D%3D)


For questions or further assistance, don’t hesitate to reach out. Let’s build together!
