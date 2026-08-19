🥭 LeafSense AI

LeafSense AI is an AI-powered mango leaf disease detection system that uses deep learning and image processing to identify diseases from mango leaf images.

🚀 Features
🌿 Mango leaf disease detection
🧠 ResNet50-based deep learning model
📷 Image upload and diagnosis
🎥 Real-time camera-based diagnosis
🔍 Grad-CAM visualization for model explainability
🖼️ ELA-based image analysis
📊 Model evaluation and performance analysis
💡 Disease information and recommended remedies
🛠️ Technologies Used
Python
TensorFlow / Keras
ResNet50
OpenCV
Tkinter
NumPy
Pandas
Matplotlib
Scikit-learn
Pillow
🧠 Model

LeafSense AI uses ResNet50, a pretrained convolutional neural network, for mango leaf disease classification.

The model uses ImageNet pretrained weights and is adapted for mango leaf disease detection.

📂 Project Structure
LeafSense-AI/
│
├── main.py
├── app_gui.py
├── advanced_evaluation.py
├── data_generator.py
├── data_loader.py
├── dataset_config.py
├── gradcam.py
├── gradcam_visualization.py
├── helper_functions.py
├── helper_setup.py
├── image_processing.py
├── imports.py
├── LeafSense_AI.spec
├── requirements.txt
│
└── utils/
    └── seed_everything.py
▶️ How to Run
1. Clone the repository
git clone https://github.com/ChaitanyaBS/LeafSense-AI.git
cd LeafSense-AI
2. Create a virtual environment
python -m venv .venv
3. Activate the virtual environment

Windows:

.venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
5. Run LeafSense AI
python main.py
📊 Model Evaluation

The project includes tools for evaluating the trained model using metrics such as:

Accuracy
F1 Score
Precision
Recall
Confusion Matrix
Precision-Recall Curve
🔍 Explainable AI

LeafSense AI includes Grad-CAM visualization to help understand which regions of a leaf image influenced the model's prediction.

📌 Project Status

🚧 Active Development

The project is being developed and improved with additional features, evaluation methods, and user-interface enhancements.

👨‍💻 Author

Chaitanya BS

GitHub: ChaitanyaBS

⭐ If you find this project useful, consider giving the repository a star!
