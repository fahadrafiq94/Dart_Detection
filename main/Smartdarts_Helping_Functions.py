import cv2
import numpy as np
import os
from google.colab.patches import cv2_imshow
import pywt
from skimage.metrics import structural_similarity as compare_ssim


def display(var_name, frame_size=None):
    if frame_size is None:
        cv2_imshow(var_name)
    else:
        resized_image = cv2.resize(var_name, frame_size)
        cv2_imshow(resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def crop_image(image, x1, y1, x2, y2):
    """
    Crop a region from the input image based on specified coordinates.

    Args:
        image (numpy.ndarray): The input image.
        x1 (int): x-coordinate of the left edge.
        y1 (int): y-coordinate of the upper edge.
        x2 (int): x-coordinate of the right edge.
        y2 (int): y-coordinate of the lower edge.

    Returns:
        numpy.ndarray: The cropped image.
    """
    if image is None:
      print("Error: Image not loaded")
    else:
      # Perform resizing operation here
      cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def extracting_roi(image, threshold=25, blur_kernel_size=9):

    """
    Arguments:
    image: The input image in which the dartboard is to be isolated.
    threshold: The threshold value for binary thresholding (default = 25).
    blur_kernel_size: The size of the kernel for Gaussian blur (default = 9).

    Output:
    isolated_dartboard: The image with only the dartboard isolated.
    """
    # Convert image to grayscale
    #grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Guassian Filtering
    blur_grayscale_gb = cv2.GaussianBlur(image, (9,9),0)

    # Apply thresholding to create a binary mask of the dartboard
    _, mask = cv2.threshold(blur_grayscale_gb, threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (the dartboard)
    # dartboard_contour = max(contours, key=cv2.contourArea)

    # Calculate contour areas
    contour_areas = [cv2.contourArea(contour) for contour in contours]


    # # For 60 degree
    # dartboard_contour = []

    # for contour in contours:
    #     # Add your criteria to select specific contours
    #     area = cv2.contourArea(contour)
    #     if area > 209 and area < 810832.5:
    #         dartboard_contour.append(contour)


    # Assuming contours_60 is already defined and contains your contours
    dartboard_contour = []
    areas_with_contours = []

    # Calculate area for each contour and store it along with the contour
    for contour in contours:
        area = cv2.contourArea(contour)
        areas_with_contours.append((area, contour))

    # Sort the list by area in descending order
    areas_with_contours.sort(key=lambda x: x[0], reverse=True)

    # Check if there are at least two contours
    if len(areas_with_contours) >= 2:
        # Get the contour with the second largest area
        second_largest_contour = areas_with_contours[1][1]

        # Append this contour to dartboard_contour_60
        dartboard_contour.append(second_largest_contour)

    # Create an empty mask for the dartboard
    dartboard_mask = np.zeros_like(mask)

    # Create a black mask with the same size as the image for 60 degree
    dartboard_mask = np.zeros(image.shape[:2], dtype="uint8")

    # Draw the contour on the mask
    # cv2.drawContours(dartboard_mask, [dartboard_contour], -1, 255, thickness=cv2.FILLED)
    # for 60 dregree
    cv2.drawContours(dartboard_mask, dartboard_contour, -1, 255, thickness=cv2.FILLED)

    # Bitwise AND the mask with the original image to isolate the dartboard
    isolated_dartboard = cv2.bitwise_and(image, image, mask=dartboard_mask)

    return isolated_dartboard, second_largest_contour

def transformation(isolated_dartboard, dartboard_corners, dst_width=400, dst_height=400, padding_factor=0.2):
    """
    Arguments:
    isolated_dartboard: The pre-processed image of the dartboard isolated from its background.
    dartboard_corners: The coordinates of the corners of the dartboard in the original image.
    dst_width (optional): The width of the destination rectangle for perspective transformation (default = 400).
    dst_height (optional): The height of the destination rectangle for perspective transformation (default = 400).
    padding_factor (optional): Factor used to adjust the corners of the destination rectangle inward (default = 0.2).

    Outputs:
    transformed_dartboard: The dartboard image after perspective transformation.
    transformed_corners: The transformed coordinates of the dartboard corners.
    center_coordinates: The calculated center coordinates of the dartboard.
    radius: The calculated radius of the dartboard.
    contours: The detected contours in the transformed dartboard image.
    """

    # Define the corners of the standardized rectangle
    dst_corners = np.array([
        [0, 0],
        [dst_width - 1, 0],
        [dst_width - 1, dst_height - 1],
        [0, dst_height - 1]
    ], dtype=np.float32)

    dartboard_corners = np.array(dartboard_corners, dtype=np.float32)
    dst_corners = np.array(dst_corners, dtype=np.float32)

    # Calculate the adjustment value
    padding_value = int(dst_width * padding_factor)

    # Adjust the destination corners slightly inward
    adjusted_dst_corners = np.array([
        [padding_value, padding_value],
        [dst_width - 1 - padding_value, padding_value],
        [dst_width - 1 - padding_value, dst_height - 1 - padding_value],
        [padding_value, dst_height - 1 - padding_value]
    ], dtype=np.float32)

    # Calculate the perspective transformation matrix using 'dartboard_corners' and 'adjusted_dst_corners'
    M = cv2.getPerspectiveTransform(dartboard_corners, adjusted_dst_corners)

    # Use perspective transformation to get the transformed coordinates of the corners
    transformed_corners = cv2.perspectiveTransform(dartboard_corners.reshape(-1, 1, 2), M)

    # Reshape the transformed corners to a flat array
    transformed_corners = transformed_corners.reshape(-1, 2)

    # Apply the perspective transformation to the original image
    transformed_dartboard = cv2.warpPerspective(isolated_dartboard, M, (dst_width, dst_height))

    # Convert the image to grayscale
    #gray_transformed_dartboard = cv2.cvtColor(transformed_dartboard, cv2.COLOR_BGR2GRAY)

    blurred_gray_transformed_dartboard = cv2.GaussianBlur(transformed_dartboard , (3, 3), 0)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(blurred_gray_transformed_dartboard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming only one contour i.e the outer part of the ellipse
    contour = contours[0]

    # Find the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate center coordinates
    center_coordinates = (int(x + w / 2), int(y + h / 2))

    # Calculate radius
    radius = int((w + h) / 4)

    return transformed_dartboard, transformed_corners, center_coordinates, radius, contours, M



def apply_wavelet_denoising(image, wavelet='db1', level=1):
    # Decompose the image using wavelet transform
    coeffs = pywt.wavedec2(image, wavelet, level=level)

    # Threshold the wavelet coefficients (remove noise)
    coeffs = [pywt.threshold(c, 0.1, mode='soft') if isinstance(c, np.ndarray) else c for c in coeffs]

    # Reconstruct the denoised image
    denoised_image = pywt.waverec2(coeffs, wavelet)

    return denoised_image.astype(np.uint8)

def preprocess(image, x1, y1, x2, y2):
    image = crop_image(image, x1, y1, x2, y2)
    #display(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image_smoothed = cv2.GaussianBlur(gray_image, (2, 2), 7, cv2.BORDER_DEFAULT)
    #image_smoothed = cv2.GaussianBlur(gray_image, (5,5), 0)
    #image_blur = cv2.medianBlur(gray_image,5)
    #gray_world_i = gray_world_assumption(gray_image)
    image_smoothed = apply_wavelet_denoising(gray_image)

    return image_smoothed



def Difference(background_image, forground_image):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    keypoints_Ix_y, descriptors_Ix_y = sift.detectAndCompute(forground_image, None)
    keypoints_Bx_y, descriptors_Bx_y = sift.detectAndCompute(background_image, None)

    # Create copies of the images for visualization
    Ix_y_vis = cv2.cvtColor(forground_image, cv2.COLOR_GRAY2BGR)
    Bx_y_vis = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR)

    # Draw keypoints on the images
    cv2.drawKeypoints(Ix_y_vis, keypoints_Ix_y, Ix_y_vis, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(Bx_y_vis, keypoints_Bx_y, Bx_y_vis, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_Ix_y, descriptors_Bx_y, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw matches
    img_matches = cv2.drawMatches(forground_image , keypoints_Ix_y, background_image , keypoints_Bx_y, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Get corresponding points
    src_pts = np.float32([keypoints_Ix_y[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_Bx_y[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find Homography matrix
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Apply Homographic transpose
    #aligned_image = cv2.warpPerspective(forground_image, M, (background_image.shape[1], background_image.shape[0]))
    aligned_image = cv2.warpPerspective(forground_image, M, (forground_image.shape[1], forground_image.shape[0]),  borderMode=cv2.BORDER_REPLICATE)

    #aligned_image = aligned_image[:background_image.shape[0], :background_image.shape[1]]
    # Apply Gaussian smoothing
    Ix_y_smooth = cv2.GaussianBlur(aligned_image, (7, 7), 0)
    Bx_y_smooth = cv2.GaussianBlur(background_image, (7, 7), 0)

    (score, diff) = compare_ssim(Bx_y_smooth, Ix_y_smooth, full=True)
    diff = (diff * 255).astype("uint8")

    #display(diff, frame_size=(500, 900))

    return diff

def Remove_noise(diff):
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # Threshold only the areas with high intensity differences (dart on the board)
    #thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY_INV)[1]

    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))
    #display(closing, frame_size=(600, 800))

    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
    #display(opening, frame_size=(600, 800))

    # Find outer contour and fill with white
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.fillPoly(opening, cnts, [255,255,255])

    #display(opening, frame_size=(600, 800))

    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(opening, kernel ,iterations = 4)

    erode = cv2.erode(dilate, kernel ,iterations = 2)
    #display(dilate, frame_size=(600, 800))

    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(erode, None, None, None, 8, cv2.CV_32S)

    #get CC_STAT_AREA component as stats[label, COLUMN]
    areas = stats[1:,cv2.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= 5000:   #keep 1500
            result[labels == i + 1] = 255

    return result

def Get_locations(image, original_image, mcont):
    img_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    dart_location = []

    # Thresholding to create a binary image
    _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # calculate the centroid of the entire mask
        M = cv2.moments(cnt)

        # ensure that the mask has a valid area before further calculations
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])

            # draw a circle at the centroid
            cv2.circle(img_copy, (centroid_x, centroid_y), 5, (0, 255, 255), -1)

            # find the point farthest from the centroid within the contour
            farthest_point = max(cnt[:, 0, :], key=lambda point: np.linalg.norm(point - [centroid_x, centroid_y]))

            # draw a circle at the farthest point
            farthest_x, farthest_y = farthest_point

            #take point within dartboard scoring contour
            dist = cv2.pointPolygonTest(mcont, (int(farthest_x), int(farthest_y)), measureDist=True)

            if dist >= 0:
              cv2.circle(img_copy, (farthest_x, farthest_y), 5, (0, 0, 255), -1)
              cv2.circle(original_image, (farthest_x, farthest_y), 5, (0, 0, 255), -1)
              dart_location.append([farthest_x, farthest_y])

        # draw contour
        cv2.drawContours(img_copy, [cnt], 0, (0, 255, 255), 2)

    # draw the bounding rectangle of the largest contour (mcont)
    #x, y, w, h = cv2.boundingRect(mcont)
    #cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 255, 102), 2)
    cv2.drawContours(img_copy, [mcont], 0, (0, 255, 255), 2)
    # display the result
    #display(img_copy, frame_size=(600, 800))
    return original_image, img_copy, dart_location


def calculate_dart_score(dart_position, transformed_dartboard, transformed_corners, center_coordinates):
    """
    Arguments:
    dart_position: The coordinates of the dart on the dartboard.
    transformed_dartboard: An image of the dartboard after perspective transformation.
    transformed_corners: The coordinates of the corners of the dartboard in the transformed image.

    Output:
    Returns the calculated score of the dart throw.
    """
    # Helper functions
    def rotate_vector(vector, angle):
        angle_rad = np.radians(angle)
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad),  np.cos(angle_rad)]])
        return np.dot(rotation_matrix, vector)

    def calculate_vector_angle(vector):
        return np.degrees(np.arctan2(vector[1], vector[0]))

    def find_score(dart_position, center_coordinates, line_angles, scores):
        radius = np.linalg.norm(np.array(dart_position) - np.array(center_coordinates))
        if radius > 169:
            return 0
        dart_vector = np.array([dart_position[0] - center_coordinates[0], dart_position[1] - center_coordinates[1]])
        dart_angle = calculate_vector_angle(dart_vector)
        dart_angle = dart_angle if dart_angle >= 0 else dart_angle + 360
        for i in range(len(line_angles)):
            start_angle = line_angles[i]
            end_angle = line_angles[(i + 1) % num_segments]
            if start_angle < end_angle:
                if start_angle <= dart_angle < end_angle:
                    return scores[i]
            else:
                if dart_angle >= start_angle or dart_angle < end_angle:
                    return scores[i]
        return 0

    # Initialize the transformed dartboard image
    contour_image = transformed_dartboard.copy()

    # corner coordinates
    # center_coordinates = center_coordinates
    first_corner = (int(transformed_corners[0][0]), int(transformed_corners[0][1]))

    # Calculate initial direction vector from the center to the first corner
    initial_direction = np.array([first_corner[0] - center_coordinates[0], first_corner[1] - center_coordinates[1]])

    # Number of segments (20 for a standard dartboard)
    num_segments = 20

    # List to store line endpoints
    line_endpoints = []

    # Rotate and draw lines, store endpoints
    for i in range(num_segments):
        rotated_direction = rotate_vector(initial_direction, 18 * i)
        end_point = (int(center_coordinates[0] + rotated_direction[0]), int(center_coordinates[1] + rotated_direction[1]))
        line_endpoints.append(end_point)
        cv2.line(contour_image, center_coordinates, end_point, (0, 255, 0), 2)

    # Calculate angles for each line segment
    line_angles = [calculate_vector_angle(np.array([end_point[0] - center_coordinates[0], end_point[1] - center_coordinates[1]])) for end_point in line_endpoints]
    line_angles = [angle if angle >= 0 else angle + 360 for angle in line_angles]

    # Assign Scores to Segments
    scores = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

    # Calculate the score with radius
    radius = np.linalg.norm(np.array(dart_position) - np.array(center_coordinates))
    if 159 <= radius <= 169:
        multiplier = 3
    elif 95 <= radius <= 105:
        multiplier = 2
    elif 7 <= radius <= 16:
        return 25  # Score is 25 for this range
    elif radius < 7:
        return 50  # Score is 50 for this range
    else:
        multiplier = 1  # Default multiplier

    return find_score(dart_position, center_coordinates, line_angles, scores) * multiplier


def transform_single_point(original_point, perspective_matrix):
    """
    Transform a single point using a perspective transformation matrix.

    Arguments:
    original_point: The original coordinates of the point (x, y).
    perspective_matrix: The perspective transformation matrix.

    Returns:
    transformed_point: The transformed coordinates of the point.
    """
    # Convert the original point to homogeneous coordinates
    original_point_homogeneous = np.array([original_point[0], original_point[1], 1], dtype=np.float32)

    # Apply the perspective transformation
    transformed_point_homogeneous = np.dot(perspective_matrix, original_point_homogeneous)

    # Convert back to Cartesian coordinates
    transformed_point = (transformed_point_homogeneous[0] / transformed_point_homogeneous[2],
                         transformed_point_homogeneous[1] / transformed_point_homogeneous[2])

    return transformed_point



import os

def read_images_and_scores(folder_path):
    image_paths = []
    scores = []
    background_image_path = None

    # List all files in the folder
    files = os.listdir(folder_path)

    # Sort files based on their numerical prefix
    files.sort(key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else float('inf'))

    for filename in files:
        if filename == "background.JPG":
            # Store the path of the background image
            background_image_path = os.path.join(folder_path, filename)
        elif filename.endswith('.jpg') or filename.endswith('.JPG'):
            # Extract score from filename
            score = int(filename.split('_')[1].split('.')[0])
            scores.append(score)

            # Store the path of the image
            image_path = os.path.join(folder_path, filename)
            image_paths.append(image_path)

    return image_paths, scores, background_image_path





