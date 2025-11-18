# 🖼️ **AI Image Classifier (Streamlit + MobileNetV2)**

This project is a simple **AI-powered image classification web app** built with **Streamlit** and **MobileNetV2**, a pre-trained deep learning model from TensorFlow trained on the ImageNet dataset.

Users can upload an image, and the app will predict the top 3 most likely objects in the image.

---

## 🚀 **Features**

* 🌐 Web-based interface using **Streamlit**
* 🤖 Uses **MobileNetV2** trained on **ImageNet (1,000 classes)**
* 📸 Upload any `.jpg` or `.png` image
* ⚡ Fast inference thanks to model caching
* 🔧 Clean, modular code (preprocessing, classification, UI separated)

---

## 🧠 **How It Works**

1. The app loads the pre-trained **MobileNetV2** model.
2. A user uploads an image.
3. The image is:
   * converted to a NumPy array
   * resized to 224×224
   * preprocessed for MobileNetV2

4. The model predicts the top 3 ImageNet labels.
5. Streamlit displays the image and predictions.

---

## 📁 **Project Structure**

```
📦 ai-image-classifier
 ┣ 📜 app.py              # Main Streamlit application
 ┣ 📜 README.md           # Project documentation
 ┣ 📜 requirements.txt    # Python dependencies
```

---

## ⚙️ **Installation & Setup**

### 1️⃣ Clone the repository

- git clone https://github.com/your-username/ai-image-classifier.git
- cd ai-image-classifier


### 2️⃣ Create a virtual environment (recommended)

- python -m venv venv
- source venv/bin/activate      # macOS/Linux
- venv\Scripts\activate         # Windows


### 3️⃣ Install dependencies

- pip install -r requirements.txt

---

## ▶️ **Run the Application**

- streamlit run app.py


Your browser will automatically open at:

http://localhost:8501

---

## 🧩 **Code Overview**

### **🔹 Model Loading**

`MobileNetV2(weights="imagenet")` loads a pre-trained deep learning model.

### **🔹 Preprocessing**

Images are resized to **224×224**, normalized, and reshaped for model input.

### **🔹 Classification**

The top 3 predictions are extracted with:

```python
decode_predictions(predictions, top=3)
```

### **🔹 UI**

Streamlit handles:

* image upload
* display
* buttons
* progress spinner

---

## 📝 **Example Output**

**Uploaded image:**
A picture of a cat.

**Predictions:**

| Label        | Confidence |
| ------------ | ---------- |
| tabby cat    | 72.3%      |
| Egyptian cat | 18.7%      |
| tiger cat    | 5.9%       |

---

## 📌 **Requirements**

* Python 3.8+
* TensorFlow 2.x
* Streamlit
* NumPy
* OpenCV
* Pillow (PIL)

---

## 📣 **Future Improvements**

* Add multiple models (ResNet, EfficientNet, Inception)
* Add webcam capture
* Improve UI with Streamlit styling
* Deploy on Streamlit Cloud
---

## 🤝 **Contributions**

Contributions, issues, and feature requests are welcome!
---

## 📄 **License**

This project is open-source under the **MIT License**.

---