import cv2
import numpy as np
import streamlit as st
import tempfile
from Smartdarts_Helping_Functions import (
    crop_image, extracting_roi, transformation, apply_wavelet_denoising,
    preprocess, Difference, Remove_noise, Get_locations,
    calculate_dart_score, transform_single_point
)
import os

def main():
    st.title('Smartdarts: Automated Dart Scoring With Computer Vision on Portable Camera')

    background_image_file = st.file_uploader("Upload background image", type=['jpg', 'png', 'jpeg'])
    dartboard_image_file = st.file_uploader("Upload dartboard image", type=['jpg', 'png', 'jpeg'])

    if background_image_file and dartboard_image_file:
        # Save uploaded images to temporary files
        with tempfile.NamedTemporaryFile(delete=False) as temp_bg_file:
            temp_bg_file.write(background_image_file.read())
            bg_filename = temp_bg_file.name
        with tempfile.NamedTemporaryFile(delete=False) as temp_dartboard_file:
            temp_dartboard_file.write(dartboard_image_file.read())
            dartboard_filename = temp_dartboard_file.name

        # Read the images using OpenCV imread
        background_image = cv2.imread(bg_filename)
        dartboard_image = cv2.imread(dartboard_filename)

       

        # Convert images to RGB format
        background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
        dartboard_image = cv2.cvtColor(dartboard_image, cv2.COLOR_BGR2RGB)


        # Perform dartboard analysis
        x1, y1, x2, y2 = 1661, 508, 2699, 1756
        background_image_segment = cv2.cvtColor(background_image, cv2.COLOR_RGB2GRAY)
        #st.image(background_image_segment, caption='Dartboard Image', use_column_width=True)
        background_image_segment = crop_image(background_image_segment, x1, y1, x2, y2)
        #st.image(background_image_segment, caption='Dartboard Image', use_column_width=True)

        isolated_dartboard, second_largest_contour = extracting_roi(background_image_segment)
        #st.image(isolated_dartboard, caption='Isolated', use_column_width=True)

        dartboard_corners = [np.array([523, 232], dtype=np.int32),
                             np.array([887, 689], dtype=np.int32),
                             np.array([628, 1067], dtype=np.int32),
                             np.array([228, 606], dtype=np.int32)]

        result = transformation(isolated_dartboard, dartboard_corners)
        transformed_dartboard, transformed_corners, center_coordinates, radius, contours, M = result

        background_image = preprocess(background_image, x1, y1, x2, y2)
        _, score_contour = extracting_roi(background_image)

        forground_image = preprocess(dartboard_image, x1, y1, x2, y2)
        diff = Difference(background_image, forground_image)
        diff = Remove_noise(diff)
        result_img, diff_thresh, locations = Get_locations(diff, forground_image, score_contour)

        st.write("Coordinates:", locations)


        transformed_coordinates = []
        for point in locations:
            # Extract x and y coordinates from the current point
            x, y = point

            # Transform the current coordinate
            transformed_point = transform_single_point((x, y), perspective_matrix=M)

            # Append the transformed coordinate to the list
            transformed_coordinates.append(transformed_point)

        # Display transformed coordinates (outside the loop)
        st.write("Transformed Coordinates:", transformed_coordinates)

        # Calculate dart scores
        dart_scores = []
        for dart_position in transformed_coordinates:
            dart_score = calculate_dart_score(dart_position, transformed_dartboard, transformed_corners, center_coordinates)
            dart_scores.append(dart_score)

        # Display dart scores (outside the loop)
        st.write("Dart Scores:", dart_scores)

        # Display the result image
        st.image(result_img, caption='Result Image')

        # Delete temporary files
        os.unlink(bg_filename)
        os.unlink(dartboard_filename)

if __name__ == '__main__':
    main()
