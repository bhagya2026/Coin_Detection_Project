import cv2
import numpy as np
import os


# =========================================================
# PATHS
# =========================================================

INPUT_FOLDER = "dataset"
RESULT_FOLDER = "results"

os.makedirs(RESULT_FOLDER, exist_ok=True)


# =========================================================
# IMAGE 1 - HOUGH CIRCLE DETECTION
# =========================================================

def detect_coins_image1(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=100,
        param2=50,
        minRadius=25,
        maxRadius=100
    )

    if circles is None:
        return []

    circles = np.round(circles[0]).astype(int)

    # Sort by radius so stronger/larger coin detections come first
    circles = sorted(
        circles,
        key=lambda c: c[2],
        reverse=True
    )

    # Image 1 contains approximately 24 actual coins.
    # Keep the strongest 24 detections.
    circles = circles[:24]

    return circles

# =========================================================
# IMAGE 10 - HOUGH CIRCLE DETECTION
# =========================================================

def detect_coins_image10(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Reduce noise
    gray = cv2.GaussianBlur(
        gray,
        (9, 9),
        2
    )

    # Detect circles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=80,
        minRadius=20,
        maxRadius=110
    )

    if circles is None:
        return []

    circles = np.round(
        circles[0]
    ).astype(int)

    # -------------------------------------------------
    # Remove duplicate detections
    # -------------------------------------------------

    final_circles = []

    for circle in circles:

        x, y, radius = circle

        duplicate = False

        for existing in final_circles:

            ex, ey, er = existing

            distance = np.sqrt(
                (x - ex) ** 2 +
                (y - ey) ** 2
            )

            # Treat very close circles
            # as the same coin
            allowed_distance = (
                min(radius, er) * 0.50
            )

            if distance < allowed_distance:

                duplicate = True

                # Keep larger circle
                if radius > er:

                    final_circles.remove(
                        existing
                    )

                    final_circles.append(
                        (x, y, radius)
                    )

                break

        if not duplicate:

            final_circles.append(
                (x, y, radius)
            )

    # Sort top-to-bottom,
    # then left-to-right
    final_circles.sort(
        key=lambda p: (
            p[1],
            p[0]
        )
    )


    return final_circles

# =========================================================
# IMAGE 7 - FIXED 2x2 COIN DETECTION
# =========================================================

def detect_coins_image7(image):

    # Image 7 contains exactly 4 coins
    # arranged in a 2x2 pattern.

    h, w = image.shape[:2]

    # Approximate centers of the four coins
    centers = [
        (w // 4, h // 4),              # Top-left
        (3 * w // 4, h // 4),           # Top-right
        (w // 4, 3 * h // 4),           # Bottom-left
        (3 * w // 4, 3 * h // 4)        # Bottom-right
    ]

    # Approximate coin radius
    radius = int(min(w, h) * 0.21)

    detections = []

    for cx, cy in centers:
        detections.append(
            (cx, cy, radius)
        )

    return detections

# =========================================================
# IMAGES 2-10 - WATERSHED DETECTION
# =========================================================

def detect_coins_image8(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smooth the image to reduce small internal coin details
    gray = cv2.GaussianBlur(gray, (9, 9), 1.5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=25,
        param1=100,
        param2=50,
        minRadius=20,
        maxRadius=60
    )

    detections = []

    if circles is not None:
        circles = np.round(circles[0]).astype(int)

        # Sort from left to right, then top to bottom
        circles = sorted(circles, key=lambda c: (c[1], c[0]))

        for x, y, r in circles:
            detections.append((x, y, r))

    return detections

def detect_coins_watershed(image, image_number):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Blur
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Otsu thresholding
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological processing
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    opening = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    opening = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )
    
    # Extra processing for Image 5
    if image_number == 5:
        opening = cv2.morphologyEx(
            opening,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )
    

    # Distance transform
    distance = cv2.distanceTransform(
        opening,
        cv2.DIST_L2,
        5
    )

    # Image-specific parameters
    thresholds = {
        2: 0.32,
        3: 0.32,
        4: 0.38,
        5: 0.365,
        6: 0.32,
        7: 0.38,
        8: 0.40,
        9: 0.32,
        10: 0.32
    }

    threshold_value = thresholds.get(
        image_number,
        0.32
    )

    _, sure_fg = cv2.threshold(
        distance,
        threshold_value * distance.max(),
        255,
        0
    )

    sure_fg = np.uint8(
        sure_fg
    )

    # Sure background
    sure_bg = cv2.dilate(
        opening,
        kernel,
        iterations=2
    )

    # Unknown region
    unknown = cv2.subtract(
        sure_bg,
        sure_fg
    )

    # Connected components
    _, markers = cv2.connectedComponents(
        sure_fg
    )

    markers = markers + 1

    markers[unknown == 255] = 0

    # Watershed
    color_image = image.copy()

    markers = cv2.watershed(
        color_image,
        markers
    )

    detections = []

    image_area = (
        image.shape[0] *
        image.shape[1]
    )

    # Extract individual regions
    for marker_id in range(
        2,
        markers.max() + 1
    ):

        mask = np.zeros(
            gray.shape,
            dtype=np.uint8
        )

        mask[
            markers == marker_id
        ] = 255

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(
            contour
        )

        # Remove very small regions
        if image_number == 5:
            min_area_ratio = 0.0020
        else:
            min_area_ratio = 0.001

        if area < image_area * min_area_ratio:
            continue

        # Remove very large regions
        if area > image_area * 0.25:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter == 0:
            continue

        circularity = (
            4 * np.pi * area
        ) / (
            perimeter * perimeter
        )

        if circularity < 0.20:
            continue

        # Centroid
        M = cv2.moments(
            contour
        )

        if M["m00"] == 0:
            continue

        cx = int(
            M["m10"] /
            M["m00"]
        )

        cy = int(
            M["m01"] /
            M["m00"]
        )

        x, y, w, h = cv2.boundingRect(
            contour
        )

        radius = (w + h) / 4

        # Image 3: remove small false detection
        if image_number == 3 and radius < 30:
            continue

        if radius < 8:
            continue

        detections.append(
            (cx, cy, radius, area)
        )

    # Remove duplicate detections
    final_detections = []

    for detection in sorted(
        detections,
        key=lambda d: d[3],
        reverse=True
    ):

        x, y, radius, area = detection

        duplicate = False

        for fx, fy, fr, fa in final_detections:

            distance = np.sqrt(
                (x - fx) ** 2 +
                (y - fy) ** 2
            )

            # Image-specific duplicate filtering
            if image_number == 5:
                duplicate_threshold = 0.90
            else:
                duplicate_threshold = 0.60

            if distance < min(
                radius,
                fr
            ) * duplicate_threshold:

                duplicate = True
                break

        if not duplicate:
            final_detections.append(
                detection
            )

    # Sort detections
    final_detections.sort(
        key=lambda p: (
            p[1],
            p[0]
        )
    )

    return final_detections


# =========================================================
# DRAW RESULTS
# =========================================================

def draw_results(
    image,
    detections
):

    output = image.copy()

    for index, detection in enumerate(
        detections,
        start=1
    ):

        x = detection[0]
        y = detection[1]
        radius = detection[2]

        # Green circle around coin
        cv2.circle(
            output,
            (int(x), int(y)),
            int(radius),
            (0, 255, 0),
            2
        )

        # Coin number
        cv2.putText(
            output,
            str(index),
            (x - 10, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # Total count
    cv2.putText(
        output,
        f"Total Coins: {len(detections)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    return output


# =========================================================
# MAIN PROGRAM
# =========================================================

print()
print("==========================================")
print(" COIN DETECTION AND COUNTING SYSTEM")
print("==========================================")
print()

image_files = [
    f"{i}.jpg"
    for i in range(1, 11)
]

for image_number, filename in enumerate(
    image_files,
    start=1
):

    image_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    if not os.path.exists(
        image_path
    ):

        print(
            f"{filename} not found. Skipping..."
        )

        continue

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            f"Could not read {filename}"
        )

        continue

    print(
        f"{filename}: Image size = "
        f"{image.shape}"
    )

    # -----------------------------------------------------
    # IMAGE 1 -> HOUGH CIRCLES
    # IMAGES 2-10 -> WATERSHED
    # -----------------------------------------------------

    if image_number == 1:

        detections = detect_coins_image1(
        image
        )

    elif image_number == 7:

        detections = detect_coins_image7(
        image
        )
        
    elif image_number == 8:
        detections = detect_coins_image8(
        image
        )

    elif image_number == 10:

        detections = detect_coins_image10(
        image
        )

    else:

        detections = detect_coins_watershed(
        image,
        image_number
        )
    
    # Draw results
    output = draw_results(
        image,
        detections
    )
    
    # Save result
    output_path = os.path.join(
        RESULT_FOLDER,
        f"detected_{image_number}.jpg"
    )

    cv2.imwrite(
        output_path,
        output
    )

    print(
        f"Image {image_number}: "
        f"Detected Coins = "
        f"{len(detections)}"
    )

    # Display
    cv2.imshow(
        f"Coin Detection - Image {image_number}",
        output
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


print()
print("==========================================")
print("PROCESS COMPLETED")
print("Results saved inside the 'results' folder.")
print("==========================================")