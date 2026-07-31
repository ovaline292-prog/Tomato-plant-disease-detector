# 🍅 Tomato Leaf Disease Classification System (Healthy vs. TYLCV)

> **GET 324: Cloud Computing & AI Model Deployment for Engineering Applications**  
> **Laboratory Exercise 10 (Mini-Project) — 15 Marks**

This repository contains an end-to-end Machine Learning pipeline and web application designed to perform automated binary pattern recognition on tomato plant leaves. Using Transfer Learning (MobileNetV2), the system accurately classifies whether a given tomato leaf is **Healthy** or infected with **Tomato Yellow Leaf Curl Virus (TYLCV)**.

---

## 📌 Course Learning Outcomes (CLOs) Addressed

* **CLO5:** Designed, trained, and evaluated a deep learning architecture (MobileNetV2 Transfer Learning) using TensorFlow/Keras on image data.
* **CLO7:** Deployed the trained CNN model as a cloud-based web application using Streamlit, managed via Git and GitHub.
* **CLO8:** Documented experimental procedures, performance metrics, and technical challenges in a structured laboratory report.

---

## 🚀 Live Demo

* **Deployed Streamlit Application:** `[INSERT YOUR STREAMLIT APP URL HERE]`
* **GitHub Repository:** `[INSERT YOUR GITHUB REPO URL HERE]`

---

## 🛠️ Project Structure

```text
├── app.py                      # Streamlit application source code
├── train_model.py              # CNN model training script (TensorFlow/Keras)
├── tomato_model.keras          # Saved trained Deep Learning model
├── requirements.txt            # Python dependencies for deployment
├── README.md                   # Project documentation
└── dataset/                    # Local training & validation images
    ├── Healthy/
    └── Yellow_Leaf_Curl/
