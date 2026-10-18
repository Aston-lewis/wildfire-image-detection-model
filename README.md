# 🔥 Wildfire Image Classification Using CNN

A deep learning project that uses a Convolutional Neural Network (CNN) to classify satellite images into two categories: **Wildfire** and **No Wildfire**. Built using Python and PyTorch, this project demonstrates how computer vision can be applied to wildfire image classification.

## 📌 About the Project

Wildfires can cause significant damage to ecosystems, wildlife, and human settlements. This project explores the use of deep learning to identify wildfire-related patterns in satellite images.

The model is trained on a Kaggle dataset containing images organized into two classes:

- **Wildfire** — Images labeled as containing wildfire.
- **No Wildfire** — Images labeled as not containing wildfire.

The project includes scripts to download the dataset, check image files for corruption or unsupported formats, train the CNN model, and classify new images.

> **Note:** This is an educational image-classification project, not a replacement for professional wildfire monitoring or emergency detection systems.

## 🛠️ Tech Stack

- **Python**
- **PyTorch** — Deep learning and model training
- **Torchvision** — Image transformations and dataset utilities
- **Pillow (PIL)** — Image processing
- **Scikit-learn** — Model evaluation metrics
- **NumPy** — Numerical operations

## 📂 Project Structure

```text
kaggle/
├── dataset/             # Downloaded dataset (not included in GitHub)
├── images/              # Example images for prediction
├── wildfire_cnn/        # Model-related folder, if generated locally
├── down.py              # Downloads the Kaggle dataset
├── check_images.py      # Checks image files for corruption or bad formats
├── train.py             # Trains the CNN model
├── predict.py           # Predicts the class of a new image
├── wildfire_cnn.pth     # Trained model weights (generated locally)
├── .gitignore
└── README.md
```

*The dataset, virtual environment, and trained model files are not included in this repository. Some folders and files may appear only after you run the scripts.*

## ⚙️ Requirements

- Python 3.10 or later, compatible with the dependencies used by this project
- Git
- A stable internet connection for downloading the dataset
- Sufficient disk space for the dataset (approximately **2 GB**) and model training
- A CUDA-compatible NVIDIA GPU is recommended for faster training, but CPU training may also be possible

## 🚀 Getting Started

### Step 1: Clone the Repository

```bash
git clone https://github.com/Aston-lewis/wildfire-image-detection-model.git
cd wildfire-image-detection-model
```

### Step 2: Create and Activate a Virtual Environment

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux:

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

Install the libraries required by the project:

```bash
pip install torch torchvision pillow numpy scikit-learn
```

Depending on your system, you may need a PyTorch installation appropriate for your CPU or CUDA version. Refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) if necessary.

## 🧠 Running the Project

Follow these steps in order.

### Step 1: Download the Dataset

The project downloads the dataset from Kaggle. The downloaded dataset occupies approximately **2 GB** of disk space.

```bash
python down.py
```

Make sure you have the necessary Kaggle access and authentication configured if the download script requires it. Review `down.py` for the dataset source and download requirements.

### Step 2: Check for Corrupted or Invalid Images

Run the image-checking script to identify images that may be corrupted or have invalid formats:

```bash
python check_images.py
```

Review the script's output before training. Follow any instructions it provides for handling problematic images.

### Step 3: Train the CNN Model

Start training:

```bash
python train.py
```

The script loads the dataset, preprocesses the images, trains the CNN, and evaluates its performance according to the implementation in `train.py`.

Training time depends on your hardware, dataset size, and training configuration. A compatible NVIDIA GPU can significantly reduce training time.

After training, the model weights are expected to be saved as:

```text
wildfire_cnn.pth
```

Check `train.py` for the exact checkpoint path and filename.

### Step 4: Classify Your Own Image

Use `predict.py` to classify an image using the trained model.

To check if you model has been trained you can run this 

```bash
python predict.py images/wild.png
```

to run your own file do this 

```bash
python predict.py /path/to/your/image.jpg
```

Replace `/path/to/your/image.jpg` with the actual path to your image.

**Example:**

```bash
python predict.py /home/your_username/Pictures/wildfire.jpg
```

The script loads the saved model, processes the supplied image, and prints the predicted class.

Possible predictions:

- `WILDFIRE`
- `NO WILDFIRE`

The prediction confidence, if displayed by the script, is a model score and should not be interpreted as a guaranteed probability of correctness.

## 📊 Model Evaluation

The project can be evaluated using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

These metrics help assess how well the model distinguishes wildfire images from non-wildfire images. Refer to the training script and its output for the actual measured results.

## ⚠️ Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- Incorrect classifications are possible, including false positives and false negatives.
- Image classification alone does not establish whether an active wildfire exists at a particular location or time.
- This model should not be used as the sole basis for emergency response or real-world fire-safety decisions.

## 🔮 Future Improvements

- Experiment with pretrained models and transfer learning.
- Improve generalization across different satellite imagery sources.
- Add a simple web interface for image uploads and predictions.
- Explore wildfire localization and segmentation in addition to image classification.

## 👨‍💻 Author

**Aston Lewis**

GitHub: [@Aston-lewis](https://github.com/Aston-lewis)

## 📄 License

This project is licensed under the MIT License if you include an MIT `LICENSE` file in the repository. Check the licenses and usage conditions of the dataset and any third-party assets separately.
