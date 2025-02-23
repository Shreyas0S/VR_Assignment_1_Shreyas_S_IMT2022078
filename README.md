# OpenCV Image Processing and Stitching

This repository contains three Python scripts utilizing OpenCV for image-processing tasks:

1. **Canny Edge Detection with Trackbars**
2. **Coin Detection and Counting**
3. **Image Stitching (Panorama Creation)**

## 1. Canny Edge Detection with Trackbars

This script applies the Canny edge detection algorithm to an image and provides trackbars to adjust threshold values dynamically.

### Features:
- Loads an image (`Coins.png`)
- Displays the original image
- Uses trackbars to control `minThreshold` and `maxThreshold`
- Shows real-time edge detection
- Saves the processed edge-detected image as `Coin_Edges.png`

### How to Run:
```bash
python canny_edge_detection.py
```

### Controls:
- Adjust the trackbars to modify edge detection sensitivity.
- Press `q` to exit the program.

---

## 2. Coin Detection and Counting

This script detects and counts coins in an image using thresholding, morphological operations, and contour detection.

### Features:
- Loads an image (`Coins.png`)
- Converts the image to grayscale
- Applies thresholding and morphological operations to segment the coins
- Uses `cv.findContours` to detect coin edges
- Counts the number of coins
- Saves intermediate and final results (`Gray_Image.png`, `Segmented_Image.png`, `Opened_Image.png`, `Closed_Image.png`, `Contours_Image.png`)
- Extracts and saves individual coins

### How to Run:
```bash
python coin_detection.py
```

### Output:
- Displays and saves images at various processing stages.
- Prints the number of detected coins in the console.

---

## 3. Image Stitching (Panorama Creation)

This script stitches two images together to create a panorama using feature detection and homography transformation.

### Features:
- Loads two images (`Left.png`, `Right.png`)
- Uses SIFT to detect keypoints and descriptors
- Matches features using FLANN-based matcher
- Computes homography transformation
- Blends overlapping regions smoothly using a blending filter
- Saves the final panorama as `Panorama.png`

### How to Run:
```bash
python image_stitching.py
```

### Output:
- Displays the input images, feature matches, and the final panorama.
- Saves the stitched panorama image.

---

## Requirements
Make sure you have the following dependencies installed before running the scripts:

```bash
pip install opencv-python numpy matplotlib
```

## Notes
- Ensure the images (`Coins.png`, `Left.png`, `Right.png`) are placed in the `../Images/` directory.
- The program requires a webcam or stored images for proper execution.
- The SIFT algorithm may require OpenCV with extra modules.


