# Pneumonia Detection Using CNN and Random Forest Ensemble
This repository contains a project that demonstrates pneumonia detection using Convolutional Neural Networks (CNN) along with an ensemble approach incorporating Random Forest. The model utilizes pretrained architectures such as VGG16, ResNet101, and Xception for feature extraction. These predictions are combined using a Random Forest classifier to achieve enhanced performance. The project employs advanced image preprocessing techniques like data augmentation and normalization for improved results.

## Table of Contents
- [Project Overview](#project-overview)
- [Published Research](#published-research)
- [Dataset](#dataset)
- [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
- [Model Architecture](#model-architecture)
- [Ensemble Approach](#ensemble-approach)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project is designed to detect pneumonia from chest X-ray images by leveraging both custom CNN models and pretrained models (VGG16, ResNet101, and Xception) to extract meaningful features. These features are then combined using a Random Forest ensemble to enhance accuracy and robustness. The model incorporates image preprocessing techniques such as data augmentation and normalization. The system’s performance was evaluated using key metrics such as accuracy, confusion matrix, and classification report, demonstrating excellent results. This project showcases the practical use of machine learning and ensemble techniques to create an automated diagnostic tool for reliable medical diagnosis.

## Published Research

📄 This project is also published as a peer-reviewed research paper:

**"Pneumonia Detection Using Machine Learning Techniques"**  
Published in: *8th International Conference on I-SMAC (IoT in Social, Mobile, Analytics and Cloud), 2024*  
🔗 [View on IEEE Xplore](https://ieeexplore.ieee.org/document/10714803)

The paper provides an in-depth explanation of the methodology, experiments, and results, highlighting the effectiveness of combining CNNs with ensemble learning for pneumonia detection.


## Dataset
The project utilizes the **Chest X-Ray Dataset**, which was collaboratively developed by researchers from Qatar University, the University of Dhaka, and their collaborators. This dataset contains labeled chest X-ray images categorized into three classes:

* **Normal**: Comprising **10,192 normal chest X-ray images** collected from multiple sources.
* **Lung Opacity (Non-COVID lung infection)**: Featuring **6,012 images** of lung opacity.
* **Viral Pneumonia**: Consisting of **1,345 viral pneumonia chest X-ray images**.

All images are in **Portable Network Graphics (PNG)** format with a resolution of **299x299 pixels**. You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database). This comprehensive dataset serves as a valuable resource for researchers aiming to develop impactful solutions for lung infections.


## Data Exploration and Preprocessing  
- **Image Preprocessing:**  
  - Data Augmentation: Rotation, scaling, and translation to avoid overfitting.  
  - Normalization: Image pixel values were normalized to a [0, 1] range.  

- **Label Encoding and Splitting:**  
  - Labels were one-hot encoded for classification.  
  - The dataset was split into **80% training** and **20% testing** using stratified sampling to maintain class balance.


## Model Architecture

The project utilizes several CNN-based models, including:

- **Custom CNN Model**: Built from scratch with BatchNormalization, Conv2D, Pooling, and Dense layers.
- **Pretrained Models**:
  - **VGG16**
  - **ResNet101**
  - **Xception**

These pretrained models were used to extract features from the X-ray images, with their final layers modified to suit the binary classification task.

## Ensemble Approach  
An **ensemble model** was created by:  
- **Neural Network Averaging:** Predictions from the custom CNN, VGG16, ResNet101, and Xception were averaged.

- **Random Forest Classifier:**  
   - A Random Forest model was trained using features extracted from VGG16.  
   - Neural network predictions were combined with Random Forest predictions to form the final ensemble output.

This ensemble approach improved the accuracy and robustness of the predictions.

## Installation

To run this project locally, follow these steps:

1. **Install Required Software:**
   - Ensure you have Python installed (preferably Python 3.6 or higher).
   - Install PyCharm (Community or Professional edition).
   - Install Git if you haven't already.
  
2. **Clone the repository**:
   
   ```bash
   git clone https://github.com/mohammadrameez/Pneumonia-Detection-Using-CNN-Ensemble-with-RF.git
   
3. **Navigate to the project directory**:
   
   ```bash
   cd pneumonia-detection-using-ml

4. **Open the Project in PyCharm**:
   
      1. **Launch PyCharm.**
      2. **Open the Project:**
         
          - Click on **Open** and select the cloned `Pneumonia-Detection-Using-CNN-Ensemble-with-RF` directory.

5. **Create a Virtual Environment (Optional but Recommended)**:
   
    1. In PyCharm, go to `File > Settings > Project: <pneumonia-detection-using-ml> > Python Interpreter`.
    2. Click on the gear icon and select **Add**.
    3. Choose **Virtualenv Environment**, select the base interpreter, and click **OK** to create a virtual environment.
    
    6. **Install Required Libraries**
    1. **Open Git Bash or the terminal in PyCharm.**
    2. **Make sure your virtual environment is activated (if you created one).**
    3. **Run the following command to install the necessary libraries:**

   ```bash
   pip install -r requirements.txt
   
6. **Manual Library Installation**
    1. If you don't have a `requirements.txt`, you can manually install the libraries using the following command:
       
          ```bash
          pip install opencv-python numpy seaborn matplotlib scikit-learn tensorflow imutils
    
Make sure these directories contain the respective images. If necessary, adjust the directory paths in the code to point to where your dataset is located.

## Usage
To train the CNN model:

1. **Prepare the Dataset**: Ensure the dataset is correctly placed in the directories:
   - `COVID-19 Radiography Database/NORMAL/`
   - `COVID-19 Radiography Database/Viral Pneumonia/`

2. **Open the Project in PyCharm**:
   - Launch PyCharm.
   - Click on **Open** and select the cloned project directory.

3. **Run the Code**:
   - In PyCharm, locate the Python script (e.g., `main.py`).
   - Right-click the file and select **Run 'main'** to execute the code.

This script will train the CNN model on the dataset and evaluate it. You will see the training output in the console, along with plots showing the training and validation loss and accuracy.

## Results
The trained CNN and Ensemble model (combining CNNs with Random Forest) achieves high accuracy on the test set for pneumonia detection, demonstrating a strong balance between sensitivity and specificity to minimize false positives and negatives. Detailed performance metrics such as accuracy, loss, and confusion matrix, along with visualizations like training/validation curves and sample predictions, are available in the results notebook for comprehensive evaluation.

## Contributing
Contributions are welcome! To contribute to this project:

1. Fork the repository.
2. Create a branch for your feature or bug fix.
3. Commit your changes.
4. Push the branch and open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE]() file for details.
