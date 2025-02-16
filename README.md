
# Multiclass Fish Image Classification

## Project Overview
This project focuses on classifying fish images into multiple categories using deep learning models. The objective is to train a CNN from scratch and leverage transfer learning with pre-trained models to enhance performance. The project also includes saving models for later use and deploying a Streamlit application to predict fish categories from user-uploaded images.

## Tech Stack & Skills
- **Deep Learning**: TensorFlow/Keras
- **Programming Language**: Python
- **Data Processing**: ImageDataGenerator
- **Transfer Learning Models**: VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0
- **Model Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Deployment**: Streamlit Web Application

## Dataset
The dataset consists of fish images categorized into folders by species. The dataset is loaded and processed using TensorFlow's ImageDataGenerator for efficient handling.

## Approach
### 1. Data Preprocessing & Augmentation
- Rescale images to [0, 1] range.
- Apply data augmentation techniques like rotation, zoom, and flipping to enhance model robustness.

### 2. Model Training
- Train a **CNN model** from scratch.
- Experiment with **five pre-trained models**: VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0.
- Fine-tune the pre-trained models on the fish dataset.
- Save the best-performing model (highest accuracy) in `.h5` format.

### 3. Model Evaluation
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrix.
- Compare performance metrics across all models to determine the most suitable architecture.
- Visualize training history (accuracy and loss) for each model separately.

### 4. Deployment
- Develop a **Streamlit web application** for real-time predictions.
- Features:
  - Allow users to upload fish images.
  - Predict and display the fish category.
  - Provide model confidence scores.

## Installation & Usage
### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn streamlit pillow
```

### Running the Project
1. Clone the repository:
```bash
[git clone https://github.com/viswakimi/Multiclass-Fish-Image-Classification]
```

2. Train the models (Optional if pre-trained models are available):
```bash
python train_models.py
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Results & Findings
- **Best Model:** The model with the highest accuracy is selected as the best-performing model.
- **Comparison Report:** Provides insights on performance across different architectures.
- **Confusion Matrix:** Visual representation of model predictions.

## Project Structure
```
|-- dataset/               # Folder containing fish images categorized by species
|-- models/                # Saved trained models (.h5 files)
|-- train_models.py        # Script to train and evaluate models
|-- app.py                 # Streamlit application script
|-- README.md              # Project documentation
```



