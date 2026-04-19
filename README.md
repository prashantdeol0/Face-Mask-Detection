# 😷 Face Mask Detection System

A real-time Face Mask Detection system built using Deep Learning and Computer Vision techniques. This project detects whether a person is wearing a mask or not through a webcam feed.

---

## 📌 Project Overview

This system uses a trained deep learning model to classify faces into two categories:

* **With Mask 😷**
* **Without Mask ❌**

It captures video from a webcam, detects faces, and predicts mask usage in real time.

---

## 🚀 Features

* 🎥 Real-time face detection using webcam
* 🧠 Deep learning model for classification
* ⚡ Fast and efficient prediction
* 🖥️ Simple UI for detection
* 📊 High accuracy model

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries & Frameworks:**

  * OpenCV
  * TensorFlow / Keras
  * NumPy
  * Matplotlib

---

## 📂 Project Structure

```
face-mask-detection/
│
├── mask_detector_model.py     # Model training script
├── mask_ui.py                 # Real-time detection script
├── mask_detector_model.h5     # Trained model (optional)
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Ignored files
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/prashantdeol0/Face-Mask-Detection
cd face-mask-detection
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ How to Run

```
python mask_ui.py
```

* Webcam will open
* The system will detect faces
* It will display:

  * **Mask 😷**
  * **No Mask ❌**

---

## 🧠 Model Details

* The model is trained using a dataset of masked and unmasked faces
* Uses Convolutional Neural Network (CNN)
* Saved as `.h5` file

---

## 📊 Future Improvements

* Improve model accuracy
* Deploy as a web application
* Add mobile support
* Use advanced architectures (ResNet, MobileNet)

---

## 📸 Output

Real-time detection showing:

* Bounding box around face
* Label (Mask / No Mask)
* Confidence score

---

## 🙋‍♂️ Author

**Prashant Deol**

* MCA Student
* Aspiring Data Scientist / AI Engineer

---

## ⭐ Show Your Support

If you like this project, please ⭐ the repository and share it!

---
