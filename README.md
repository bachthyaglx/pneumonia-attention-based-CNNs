# Introduction
Pneumonia is a serious lung infection that can be life-threatening, especially in places with limited medical care. Detecting it early is very important for successful treatment. Doctors often use chest X-ray images to diagnose pneumonia, but reading these images can be difficult because the infection can look different in each patient.
This project uses deep learning to improve pneumonia detection by applying an attention-based Convolutional Neural Network (CNN). Traditional CNNs are good at identifying patterns in images, but they sometimes miss important details because they focus on the whole image instead of key areas. Attention-based CNNs help solve this problem by highlighting the most important parts of the X-ray, making the diagnosis more accurate.

# Dataset
https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia

# Methodology
The methodology of this project is based on deep learning techniques for classifying chest X-ray images into three categories: Normal, Pneumonia, and COVID-19. The implementation is divided into two main approaches:
* Traditional CNN-Based Classification (cxr_classification.ipynb)
* Attention-Based CNN Classification (cxr_classification_attentionCNN.ipynb)

The following steps outline the methodology applied in both approaches:

1. Data Preprocessing
* Resizing: All images are resized to a fixed dimension (e.g., 224x224 or 256x256 pixels) to maintain consistency across the dataset.
* Normalization: Pixel values are scaled between 0 and 1 to improve training stability.
* Data Augmentation: To increase the diversity of training samples, techniques such as rotation, flipping, and zooming are applied.

2. Model Architectures

A. Traditional CNN Model (cxr_classification.ipynb)
* Convolutional Layers: Extracts important image features like edges and textures.
* Batch Normalization & ReLU Activation: Normalizes the data and introduces non-linearity.
* Max-Pooling Layers: Reduces feature map size while retaining essential information.
* Fully Connected Layers (Dense): Transforms extracted features into a classification decision.
* Softmax Activation: Outputs probability scores for each category (Normal, Pneumonia, or COVID-19).

B. Attention-Based CNN Model (cxr_classification_attentionCNN.ipynb)
* Convolutional Layers: Extract local features.
* Spatial and Channel Attention Modules: Helps the model focus on the most important areas of the image.
* Batch Normalization & ReLU Activation: Standardization and non-linearity.
* Max-Pooling Layers: Reduces feature dimensions while keeping relevant information.
* Fully Connected Layers (Dense): Converts feature maps into classification predictions.
* Softmax Activation: Generates the final category probabilities.

3. Model Training and Optimization
Both models are trained using supervised learning with labeled X-ray images. The training process follows these key steps:
* Loss Function: Categorical Cross-Entropy is used as the loss function since the task is multi-class classification.
* Optimizer: Adam optimizer is selected for efficient gradient updates.
* Learning Rate Scheduling: A scheduler adjusts the learning rate to improve convergence.
* Early Stopping: Stops training if validation accuracy does not improve after multiple epochs to prevent overfitting.

5. Model Evaluation
After training, both models are evaluated using the test dataset to measure performance. The following metrics are calculated:
* Accuracy: Measures overall correctness of predictions.
* Precision, Recall, and F1-Score: Evaluates how well the model identifies pneumonia and COVID-19 cases.
* Confusion Matrix: Visualizes classification errors and correct predictions.
* ROC Curve & AUC Score: Assesses the model’s ability to distinguish between classes.

5. Comparison Between Traditional CNN and Attention-Based CNN
* Traditional CNN: Captures general image features but treats all regions equally, potentially missing critical patterns.
* Attention-Based CNN: Dynamically focuses on important areas (such as infected lung regions), leading to better classification accuracy.

By integrating the attention mechanism, the model improves its ability to detect pneumonia and COVID-19 more effectively, making it a more robust and interpretable approach for medical image analysis.

# Evaluation and Results

# Guide how to run the project
To run this project on your local machine, follow these steps:

1. Install Dependencies

Ensure you have Python 3.x installed on your system. Then, install the required libraries using the requirements.txt file.

`pip install -r requirements.txt`

2. Download the Dataset

The dataset used in this project is Chest X-ray COVID-19 Pneumonia Dataset from Kaggle.
* Download it from this Kaggle link.
* Extract the dataset into the project folder. Make sure the dataset structure looks like this:

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

Option 1 - Run the Jupyter Notebook
Open and run either of the following notebooks:
* cxr_classification.ipynb → Runs a traditional CNN model
* cxr_classification_attentionCNN.ipynb → Runs an attention-based CNN model


Option 2 - Running on Google Colab
* Upload dataset to google colab
* Run both ipynb files

