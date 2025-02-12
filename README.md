# Introduction
Pneumonia is a serious lung infection that can be life-threatening, especially in places with limited medical care. Detecting it early is very important for successful treatment. Doctors often use chest X-ray images to diagnose pneumonia, but reading these images can be difficult because the infection can look different in each patient.
This project uses deep learning to improve pneumonia detection by applying an attention-based Convolutional Neural Network (CNN). Traditional CNNs are good at identifying patterns in images, but they sometimes miss important details because they focus on the whole image instead of key areas. Attention-based CNNs help solve this problem by highlighting the most important parts of the X-ray, making the diagnosis more accurate.

# Dataset
https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia

# Methodology
The methodology of this project is based on deep learning techniques for classifying chest X-ray images into three categories: Normal, Pneumonia, and COVID-19. The implementation is divided into two main approaches:
	1.	Traditional CNN-Based Classification (cxr_classification.ipynb)
	2.	Attention-Based CNN Classification (cxr_classification_attentionCNN.ipynb)

The following steps outline the methodology applied in both approaches:

1. Data Preprocessing

The dataset consists of chest X-ray images categorized into Normal, Pneumonia, and COVID-19 cases. Before training the model, the images go through several preprocessing steps to ensure better feature extraction and model performance:
	•	Resizing: All images are resized to a fixed dimension (e.g., 224x224 or 256x256 pixels) to maintain consistency across the dataset.
	•	Normalization: Pixel values are scaled between 0 and 1 to improve training stability.
	•	Data Augmentation: To increase the diversity of training samples, techniques such as rotation, flipping, and zooming are applied.

2. Model Architectures

A. Traditional CNN Model (cxr_classification.ipynb)

A standard Convolutional Neural Network (CNN) architecture is implemented to extract patterns from chest X-ray images. The architecture follows these layers:
	1.	Convolutional Layers: Extracts important image features like edges and textures.
	2.	Batch Normalization & ReLU Activation: Normalizes the data and introduces non-linearity.
	3.	Max-Pooling Layers: Reduces feature map size while retaining essential information.
	4.	Fully Connected Layers (Dense): Transforms extracted features into a classification decision.
	5.	Softmax Activation: Outputs probability scores for each category (Normal, Pneumonia, or COVID-19).

B. Attention-Based CNN Model (cxr_classification_attentionCNN.ipynb)

To improve classification accuracy, an Attention Mechanism is integrated into the CNN. This helps the model focus on critical regions of the X-ray images rather than treating all parts of the image equally. The modified architecture includes:
	1.	Convolutional Layers: Extract local features.
	2.	Spatial and Channel Attention Modules: Helps the model focus on the most important areas of the image.
	3.	Batch Normalization & ReLU Activation: Standardization and non-linearity.
	4.	Max-Pooling Layers: Reduces feature dimensions while keeping relevant information.
	5.	Fully Connected Layers (Dense): Converts feature maps into classification predictions.
	6.	Softmax Activation: Generates the final category probabilities.

3. Model Training and Optimization

Both models are trained using supervised learning with labeled X-ray images. The training process follows these key steps:
	•	Loss Function: Categorical Cross-Entropy is used as the loss function since the task is multi-class classification.
	•	Optimizer: Adam optimizer is selected for efficient gradient updates.
	•	Learning Rate Scheduling: A scheduler adjusts the learning rate to improve convergence.
	•	Early Stopping: Stops training if validation accuracy does not improve after multiple epochs to prevent overfitting.

4. Model Evaluation

After training, both models are evaluated using the test dataset to measure performance. The following metrics are calculated:
	•	Accuracy: Measures overall correctness of predictions.
	•	Precision, Recall, and F1-Score: Evaluates how well the model identifies pneumonia and COVID-19 cases.
	•	Confusion Matrix: Visualizes classification errors and correct predictions.
	•	ROC Curve & AUC Score: Assesses the model’s ability to distinguish between classes.

5. Comparison Between Traditional CNN and Attention-Based CNN
	•	Traditional CNN: Captures general image features but treats all regions equally, potentially missing critical patterns.
	•	Attention-Based CNN: Dynamically focuses on important areas (such as infected lung regions), leading to better classification accuracy.

By integrating the attention mechanism, the model improves its ability to detect pneumonia and COVID-19 more effectively, making it a more robust and interpretable approach for medical image analysis.

# Evaluation and Results

# Guide how to run the project
To run this project on your local machine, follow these steps:

1. Install Dependencies

Ensure you have Python 3.x installed on your system. Then, install the required libraries using the requirements.txt file.

Using pip

Open a terminal or command prompt and run:

pip install -r requirements.txt

If you don’t have a requirements.txt file, manually install the required dependencies:

pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python

2. Download the Dataset

The dataset used in this project is Chest X-ray COVID-19 Pneumonia Dataset from Kaggle.
	1.	Download it from this Kaggle link.
	2.	Extract the dataset into the project folder. Make sure the dataset structure looks like this:

/pneumonia-attention-based-CNNs
├── dataset/
│   ├── Normal/
│   ├── Pneumonia/
│   ├── COVID-19/
│   └── metadata.csv
├── cxr_classification.ipynb
├── cxr_classification_attentionCNN.ipynb
├── requirements.txt
└── README.md

3. Run the Jupyter Notebook

To execute the project, use Jupyter Notebook:
	1.	Open a terminal and navigate to the project directory:

cd pneumonia-attention-based-CNNs


	2.	Start Jupyter Notebook:

jupyter notebook


	3.	Open and run either of the following notebooks:
	•	cxr_classification.ipynb → Runs a traditional CNN model
	•	cxr_classification_attentionCNN.ipynb → Runs an attention-based CNN model

4. Train the Model

Inside the notebook:
	1.	Run all cells sequentially to preprocess data, build the model, train, and evaluate.
	2.	The training process may take several minutes to hours depending on your hardware.
	3.	Training metrics such as accuracy and loss will be displayed in real time.

5. Evaluate the Model
	•	Once training is complete, the notebook will display the classification results using metrics such as accuracy, precision, recall, and confusion matrix.
	•	You can also visualize sample predictions and feature maps to understand model performance.

6. Save and Load the Model (Optional)

To save the trained model, use:

model.save("pneumonia_model.h5")

To load a pre-trained model and use it for prediction, run:

from tensorflow.keras.models import load_model
model = load_model("pneumonia_model.h5")

7. Running on Google Colab (Optional)

If you don’t have a GPU on your local machine, you can run the project on Google Colab:
	1.	Upload the project files to Google Drive.
	2.	Open Google Colab and mount your drive:

from google.colab import drive
drive.mount('/content/drive')


	3.	Change the working directory to the project folder:

%cd /content/drive/MyDrive/pneumonia-attention-based-CNNs


	4.	Run the notebook cells as usual.

8. Inference (Making Predictions on New X-rays)

After training, you can use the model to classify new chest X-ray images:

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("pneumonia_model.h5")

# Load and preprocess new image
image_path = "path_to_new_xray.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (224, 224))  # Resize to match model input
img = img / 255.0  # Normalize pixel values
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Make prediction
prediction = model.predict(img)
print("Prediction:", prediction)

9. Troubleshooting
	•	If Jupyter Notebook does not start, install Jupyter:

pip install notebook


	•	If the model training takes too long, use a GPU (Google Colab or a local CUDA-enabled GPU).
	•	If an error occurs due to missing dependencies, reinstall the required libraries using:

pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python.
