# 🔍 Fingerprint Classification – ML Model - Partial Fingerprint Recognizer 

Welcome to the public portion of project **Half Mark**. This repository focuses specifically on the **model training** and **inference (prediction)** stages of the system, using deep learning techniques for fingerprint classification and partial fingerprint recognition.

> 🛡️ **Note**: The complete project includes a full-stack interface, extended preprocessing pipeline, and user interaction modules which are **private** and can be shared **upon request**.

---

## 📌 Project Summary

The core objective of this project is to classify fingerprint images into one of the following major fingerprint pattern types:

- **Loops**
- **Whorls**
- **Arches**
- *(Other minor types if applicable)*

To accomplish this, we utilized **Transfer Learning** with the **ResNet50** architecture, leveraging its power to extract deep spatial features from fingerprint images.

---

## 🧠 Model Details

- 📚 **Architecture**: ResNet50 (pre-trained on ImageNet, fine-tuned on fingerprint data)
- 🛠️ **Framework**: TensorFlow / Keras
- 🗂️ **Classes**: Loops, Whorls, Arches etc
- 🧪 **Training**: Custom fingerprint dataset with data preprocessing techniques
- 💾 **Output**: A `.h5` file containing the trained model — **ready for deployment**

---

## 📁 What’s Included in This Repo

- ✅ Preprocessing logic for input images
- ✅ ResNet50-based model definition and training code
- ✅ Script for running predictions on new fingerprint images
- ✅ The **trained `.h5` model file** for instant use — no need to retrain unless desired

---

## ❌ What’s Not Included (But Available on Demand)

The following components are **not published** on this public repo, but are available upon request:

- Full **frontend UI**
- **Web or desktop app interface**
- Extended data cleaning, segmentation, and enhancement pipeline
- Additional documentation, deployment scripts, and backend integration

---

## 🖼️ Home Screen (UI Preview)

A screenshot of the home screen from the private part of the project is included below to give a visual sense of the complete system:

![Home Screen Preview](./assets/home_screen.png)  

---
