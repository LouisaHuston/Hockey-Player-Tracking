
# Hockey Player Tracking

This project focuses on tracking hockey players in game footage, utilizing machine learning models and various data processing techniques to identify and monitor players' movements and positions.

## Project Structure

- **train.py**: Main script to train the tracking model on hockey game data.
- **test.py**: Script to test and evaluate the trained model.
- **inference.py**: Runs the trained model on new data to generate tracking predictions.
- **requirements.txt**: Lists all dependencies required to run the project.
- **process_data.py**: A data processing script that may complement or replace some functions in `src/process.py`.
- **src/**: Contains all the core modules for data processing, model definition, and evaluation.
  - `download_data.py`: Handles the download of necessary datasets.
  - `process.py`: Prepares and processes raw data for training and inference.
  - `detr.py`: Implementation of a DEtection TRansformer (DETR) model for advanced tracking and object detection tasks.
  - `coco_dataset.py`: Defines dataset handling, particularly for COCO-format data, which may be used to train and validate models.
  - `evaluation.py`: Evaluates model performance on test data.
  - `split.py`: Handles data splitting, potentially to separate training, validation, and test sets.

## Features

- **Data Processing**: Automated data download and preprocessing for efficient training.
- **Model Definition**: Comprehensive model architecture specifically optimized for tracking hockey players, with additional support for the DETR model.
- **Evaluation and Testing**: Tools to test and evaluate model accuracy and effectiveness on new data.

## Getting Started

### Prerequisites

Ensure you have Python 3.8 or higher installed.

### Virtual Environment Setup

It is recommended to set up a virtual environment for managing dependencies. You can create a virtual environment named `hockey` and install the dependencies as follows:

```bash
# Create a virtual environment
python3 -m venv hockey

# Activate the virtual environment
# On macOS/Linux:
source hockey/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

1. **Data Download**: Use `src/download_data.py` to download the necessary datasets.
2. **Data Processing**: Run `process_data.py` or `src/process.py` to preprocess the data and prepare it for model training.
3. **Training**: Execute `train.py` to train the model on the hockey player tracking dataset.
4. **Inference**: Run `inference.py` to use the trained model on new video data.

### Example Usage

To train the model with the default settings:

```bash
python train.py
```

To perform inference on a new dataset:

```bash
python inference.py
```

### Directory Structure

```
Hockey-Player-Tracking/
├── src/
│   ├── download_data.py
│   ├── process.py
│   ├── detr.py
│   ├── coco_dataset.py
│   ├── evaluation.py
│   └── split.py
├── train.py
├── test.py
├── inference.py
├── process_data.py
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any feature additions or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
