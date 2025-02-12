# Pneumonia Attention-Based CNNs

Aribah Ibrahim, 
Seyedmostafa Musavi, 
Khuu Bach Thy

# Introduction
Pneumonia is a serious lung infection that can be life-threatening, especially in places with limited medical care. Detecting it early is very important for successful treatment. Doctors often use chest X-ray images to diagnose pneumonia, but reading these images can be difficult because the infection can look different in each patient.
This project uses deep learning to improve pneumonia detection by applying an attention-based Convolutional Neural Network (CNN). Traditional CNNs are good at identifying patterns in images, but they sometimes miss important details because they focus on the whole image instead of key areas. Attention-based CNNs help solve this problem by highlighting the most important parts of the X-ray, making the diagnosis more accurate.

# Dataset
### Dataset: Chest X-Ray Images (COVID-19 & Pneumonia)
This dataset contains chest X-ray images for **COVID-19**, **Pneumonia**, **Lung Opacity** and **Normal** cases. It is used in this project to train and evaluate an **Attention-Based Convolutional Neural Network (CNN)** for the detection and classification of pneumonia and COVID-19 from chest X-ray images.

![image](https://github.com/user-attachments/assets/4a165652-7ac3-432b-bab9-80d9f91fb05d)

#### Dataset Overview:
- **Source**: [Kaggle](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)
- **Total Images**: ~6,000+ (across all categories)
- **Image Type**: Chest X-ray (JPEG format)
- **Classes**:
  - **COVID-19**: X-ray images of patients diagnosed with COVID-19.
  - **Pneumonia**: X-ray images of patients diagnosed with pneumonia.
  - **Lung Opacity**: X-ray images of patients diagnosed with lung.
  - **Normal**: X-ray images of healthy individuals with no signs of disease.
  
![image](https://github.com/user-attachments/assets/66238d9e-4875-485f-bd21-406494477c49)

#### Dataset Usage in This Project:
- The dataset is preprocessed and used to train an **Attention-Based CNN** model for pneumonia detection.
- The model leverages attention mechanisms to focus on relevant regions of the chest X-ray images, improving classification accuracy.
- The dataset is split into training, validation, and test sets to evaluate the model's performance.

# Methodology
The methodology of this project is based on deep learning techniques for classifying chest X-ray images into three categories: Normal, Pneumonia, and COVID-19.
* Attention-Based CNN Classification (cxr_classification_attentionCNN.ipynb)

The following steps outline the methodology applied in both approaches:

## 1. Data Preprocessing
* Resizing: All images are resized to a fixed dimension of 224x224 to maintain consistency across the dataset.
* Normalization: Pixel values are scaled between 0 and 1 to improve training stability.
* Data Augmentation: To increase the diversity of training samples, techniques such as rotation, flipping, and zooming are applied.

## 2. Model Architectures
Attention-Based CNN Model (cxr_classification_attentionCNN.ipynb)

![image](https://github.com/user-attachments/assets/10bddc85-8980-478d-bdef-4f58ccdf0127)

* Convolutional Layers: Extract local features.
* Spatial and Channel Attention Modules: Helps the model focus on the most important areas of the image.
* Batch Normalization & ReLU Activation: Standardization and non-linearity.
* Max-Pooling Layers: Reduces feature dimensions while keeping relevant information.
* Fully Connected Layers (Dense): Converts feature maps into classification predictions.
* Softmax Activation: Generates the final category probabilities.

## 3. Model Training and Optimization
Both models are trained using supervised learning with labeled X-ray images. The training process follows these key steps:
* Loss Function: Categorical Cross-Entropy is used as the loss function since the task is multi-class classification.
* Optimizer: Adam optimizer is selected for efficient gradient updates.
* Learning Rate Scheduling: A scheduler adjusts the learning rate to improve convergence.
* Early Stopping: Stops training if validation accuracy does not improve after multiple epochs to prevent overfitting.

## 4. Model Evaluation
After training, both models are evaluated using the test dataset to measure performance. The following metrics are calculated:
* Accuracy: Measures overall correctness of predictions.
* Precision, Recall, and F1-Score: Evaluates how well the model identifies pneumonia and COVID-19 cases.
  
  ![image](https://github.com/user-attachments/assets/149c671b-9781-4259-9e7e-79295e10a0c3)

  ![image](https://github.com/user-attachments/assets/7874e986-d8f5-48b9-b42a-0251df5dee5f)
  
* Confusion Matrix: Visualizes classification errors and correct predictions.

![image](https://github.com/user-attachments/assets/1c7919f1-2f8c-4eb4-9754-a8f5287c1c25)

# Guide how to run the project
To run this project on your local machine, follow these steps:

## 1. Install Dependencies
`pip install -r requirements.txt`

## 2. Download the Dataset
The dataset used in this project is Chest X-ray COVID-19 Pneumonia Dataset from Kaggle.
* Download it from this Kaggle link.
* Extract the dataset into the project folder. Make sure the dataset structure looks like this:
  
## 3. Run project
### Option 1 - Run the Jupyter Notebook
Open and run either of the following notebooks:
* cxr_classification_attentionCNN.ipynb → Runs an attention-based CNN model

### Option 2 - Running on Google Colab

### i. Upload the Dataset to Google Colab
- Download the dataset from [Kaggle](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia).
- Open the notebook in Google Colab using the links below.
- In Colab, use the **Files** sidebar to upload the dataset:
  1. Click the **Files** icon on the left sidebar.
  2. Click **Upload to session storage**.
  3. Select the dataset folder (`chest_xray_covid19_pneumonia`) and upload it.

### ii. Run the Notebooks
- Click the links below to open the notebooks in Google Colab:
  - [Notebook 1: Preprocessing and Training](https://colab.research.google.com/github/bachthyaglx/pneumonia-attention-based-CNNs/blob/main/notebook1.ipynb)
  - [Notebook 2: Evaluation and Visualization](https://colab.research.google.com/github/bachthyaglx/pneumonia-attention-based-CNNs/blob/main/notebook2.ipyn

# Reference
[Attention-based Convolutional Neural Network](https://medium.com/@clairenyz/attention-based-convolutional-neural-network-a719693058a7)



