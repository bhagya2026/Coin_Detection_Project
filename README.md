# 🪙 Coin Detection and Counting System

An image processing-based system that detects and counts multiple coins from an input image using **Python and OpenCV**.

## 📌 Project Overview

The **Coin Detection and Counting System** is a Digital Image Processing project designed to automatically detect coins present in an image and calculate the total number of detected coins.

The system uses image processing techniques such as:

- Grayscale conversion
- Gaussian blurring
- Thresholding
- Morphological operations
- Distance transformation
- Watershed segmentation
- Hough Circle Detection

Different detection techniques are used according to the characteristics of the input image.

## 🎯 Objectives

- Detect multiple coins from an image.
- Separate coins that are close to each other.
- Count the detected coins automatically.
- Display the detected coins with numbered labels.
- Generate processed output images.
- Improve detection for different image conditions.

## ✨ Features

- 🪙 Automatic coin detection
- 🔢 Automatic coin counting
- 📌 Numbering of detected coins
- 🖼️ Image-based processing
- 🔍 Hough Circle Detection
- 🌊 Watershed segmentation
- 📊 Detection results saved as output images
- ⚙️ Image-specific detection optimization

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| OpenCV | Image processing and computer vision |
| NumPy | Numerical operations |
| Git | Version control |
| GitHub | Project repository |

## 📂 Project Structure

```text
Coin_Detection_Project/
│
├── dataset/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── ...
│   └── 10.jpg
│
├── results/
│   ├── detected_1.jpg
│   ├── detected_2.jpg
│   ├── ...
│   └── detected_10.jpg
│
└── main.py
```

## ⚙️ How It Works

The system follows these general steps:

```text
Input Image
     ↓
Preprocessing
     ↓
Grayscale Conversion
     ↓
Noise Reduction
     ↓
Image Segmentation
     ↓
Coin Detection
     ↓
Duplicate/Invalid Detection Removal
     ↓
Coin Counting
     ↓
Numbered Output Image
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/bhagya2026/Coin_Detection_Project.git
```

### 2. Open the project folder

```bash
cd Coin_Detection_Project
```

### 3. Install the required libraries

```bash
pip install opencv-python numpy
```

### 4. Run the program

```bash
python main.py
```

The processed images will be saved in the:

```text
results/
```

folder.

## 📊 Current Detection Status

The current version has been tested on 10 sample images.

| Image | Detected Coins | Status |
|---|---:|---|
| 1 | 24 | ✅ Working |
| 2 | 4 | ✅ Working |
| 3 | 12 | ✅ Working |
| 4 | 11 | 🔧 Further improvement |
| 5 | 15 | 🔧 Further improvement |
| 6 | 10 | ✅ Working |
| 7 | 4 | ✅ Working |
| 8 | 6 | ✅ Working |
| 9 | 6 | ✅ Working |
| 10 | 8 | ✅ Working |

**Note:** Images 4 and 5 are still being improved for better detection accuracy.

## 📸 Output

The detected coins are marked and numbered in the generated output images.

The total number of detected coins is also displayed on the output image.

## 🔮 Future Improvements

- Improve detection of overlapping coins.
- Improve detection under different lighting conditions.
- Handle different backgrounds.
- Improve detection of partially visible coins.
- Support coin denomination recognition.
- Calculate the total monetary value of detected coins.
- Improve accuracy using machine learning/deep learning techniques.
- Support real-time coin detection using a camera.

## 👩‍💻 Author

**Bhagyashree B**

MCA Student

## 📄 Project Type

**Digital Image Processing / Computer Vision Project**

---

⭐ If you find this project useful, consider giving the repository a star.
