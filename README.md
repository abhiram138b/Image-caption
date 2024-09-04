# Image Captioning Project

This repository contains the code for an image captioning model that uses CNN (VGG16) for feature extraction and LSTM for sequence generation.

## Project Structure

- `data/`: Contains the dataset and preprocessed vocabulary.
- `models/`: Holds the model definition and training scripts.
- `utils/`: Contains helper scripts for data loading and evaluation.
- `results/`: Stores the output results such as loss plots and generated captions.
- `README.md`: Overview of the project and instructions.
- `requirements.txt`: Python dependencies required to run the project.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Torchvision
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```
    git clone https://github.com/your_username/Image-Captioning-Project.git
    cd Image-Captioning-Project
    ```
2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

### Usage

1. Prepare the dataset and place it in the `data/Flickr8k/` folder.
2. Run the training script:
    ```
    python models/train.py
    ```
3. Check results in the `results/` folder.

### Model

The model consists of a CNN using VGG16 for feature extraction and an LSTM for generating image captions.

### Results

- Achieved a low cross-entropy loss of 1.09, demonstrating the model's ability to generate accurate and meaningful captions.
