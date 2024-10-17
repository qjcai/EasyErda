import cv2
import numpy as np
import pytesseract
from scipy.spatial import distance


def flame_process(user_image):
    # Load the image
    image = cv2.imread(user_image)

    # 1. Scale the Image
    scale_factor = 2
    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # 2. Apply Less Aggressive Sharpening
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])

    sharpened = cv2.filter2D(resized_image, -1, sharpening_kernel)

    # 3. Convert the Image to Grayscale
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

    # Save and check the sharpened and grayscale image (optional)
    cv2.imwrite('debug_resized_sharpened_gray.png', gray)

    original_text = pytesseract.image_to_string(gray, lang='eng')

    # 4. Use pytesseract to do OCR on the processed image
    text = original_text.replace('#', '+')
    text = text.replace('potentiat', 'Potential')
    # print(text)

    # 5. Print the extracted text
    # print("Extracted Text:")

    # Convert the text to lowercase for case-insensitive search
    lower_text = text.lower()

    # Define the keywords and convert them to lowercase
    start_keyword = "type".lower()
    end_keyword = "potential".lower()

    # Find the starting and ending positions
    start_index = lower_text.find(start_keyword)
    end_index = lower_text.find(end_keyword) - 2

    # Extract and print the relevant section of the text
    if start_index != -1:

        # 6. Get the bounding boxes for each detected text element
        d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        # print(d)
        n_boxes = len(d['text'])
        padding = 2
        for i in range(n_boxes):
            if int(d['conf'][i]) > 60:  # Filter out low-confidence detections
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])

                # Apply padding to make the box bigger
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = w + 2 * padding
                h = h + 2 * padding

                # resized_image = cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

                # Put the detected text above the bounding box
                text = d['text'][i]
                font_scale = 0.5
                thickness = 1
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_x = x
                text_y = y - 5 if y - 5 > 5 else y + h + text_size[1] + 5

        # 7. Display the image with bounding boxes
        # cv2.imshow('Original', resized_image)

        # Save the image with bounding boxes (optional)
        # cv2.imwrite('output_with_boxes.png', resized_image)

        # Define the target color (e.g., a specific shade of blue in RGB)
        target_color = np.array([204, 255, 0])

        # Define the color range for detection
        color_range = 50
        lower_bound = target_color - color_range
        upper_bound = target_color + color_range

        # Extract bounding box coordinates for the keywords
        if start_index != -1 and end_index != -1:  # and end_index > start_index:
            # Get bounding box of the entire region from 'Type' to 'Potential'
            d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            n_boxes = len(d['text'])
            x1, y1, x2, y2 = None, None, None, None
            # d = d['text'].replace('fype', 'Type')
            # print(d['text'])
            for i in range(n_boxes):
                if start_keyword in d['text'][i].lower() or end_keyword in d['text'][i].lower():
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h
                    break

            if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                # Increase padding around the bounding box
                padding = 250
                y_padding = 110
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(resized_image.shape[1], x2 + padding), min(resized_image.shape[0], y2 + padding + y_padding)

                # Debug: Print bounding box coordinates
                # print(f"Adjusted Bounding Box Coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                # Extract ROI based on the keyword bounding box
                roi_image = resized_image[y1:y2, x1:x2]
                cv2.imwrite('gogogo.png', roi_image)
                if roi_image.size == 0:
                    # print(x1, x2, y1, y2, roi_image.size)
                    print("ROI is empty.")
                else:
                    # print(x1, x2, y1, y2, roi_image.size)

                    image_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite('output_image_rgb.png', image_rgb)
                    roi_image = image_rgb[y1:y2, x1:x2]
                    # print(image_rgb.size)
                    # Debug: Show the ROI
                    # cv2.imshow('ROI', image_rgb)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # Apply color mask to the ROI
                    roi_mask = cv2.inRange(image_rgb, lower_bound, upper_bound)

                    # Debug: Show the Mask
                    # cv2.imshow('Mask', roi_mask)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # Apply the mask to the ROI to get the color-matched region
                    roi_colored = cv2.bitwise_and(image_rgb, image_rgb, mask=roi_mask)

                    # Debug: Show the Colored Region
                    # cv2.imshow('Colored ROI', roi_colored)
                    cv2.imwrite('output_roi_colored.png', roi_colored)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # Convert the masked ROI to grayscale
                    roi_gray = cv2.cvtColor(roi_colored, cv2.COLOR_BGR2GRAY)
                    # Debug: Show the Grayscale ROI
                    # cv2.imshow('Gray ROI', roi_gray)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # Check if the grayscale image is empty
                    if np.count_nonzero(roi_gray) == 0:
                        print(
                            "The grayscale image is completely black, indicating no text was detected in the target color range.")

                    # Use pytesseract to extract text from the color-matched ROI
                    color_matched_text = pytesseract.image_to_string(roi_gray, lang='eng')
                    # print("Text in Color-Matched Region:")
                    # print(color_matched_text)
            else:
                print("Keywords not found or bounding box is invalid.")

        else:
            print(f"'{start_keyword}' or '{end_keyword}' not found in the text.")
            return

        ###########

        image = cv2.imread('output_roi_colored.png')

        # 1. Scale the Image
        scale_factor = 2
        resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # 2. Apply Less Aggressive Sharpening
        sharpening_kernel = np.array([[0, -1, 0],
                                      [-1, 5, -1],
                                      [0, -1, 0]])

        sharpened = cv2.filter2D(resized_image, -1, sharpening_kernel)

        # 3. Convert the Image to Grayscale
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

        # Save and check the sharpened and grayscale image (optional)
        # cv2.imwrite('debug_resized_sharpened_gray.png', gray)

        # Load the image
        image = cv2.imread('output_roi_colored.png')

        if image is None:
            raise ValueError("Image not found or unable to load.")

        # Convert the image to HSV (Hue, Saturation, Value) color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the target color (e.g., bright green) in HSV space
        # You can adjust the ranges to better fit the target color
        lower_bound = np.array([30, 100, 100])  # Example: Lower bound for green
        upper_bound = np.array([90, 255, 255])  # Example: Upper bound for green

        # Create a mask that isolates the target color
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Define a distance threshold (adjust this value as needed)
        distance_threshold = 23

        # Store the bounding boxes
        bounding_boxes = []

        # Iterate through each contour
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            new_box = (x, y, x + w, y + h)
            new_center = ((x + x + w) / 2, (y + y + h) / 2)

            # Check if the new box is close to existing boxes
            added_to_existing = False
            for i, box in enumerate(bounding_boxes):
                box_x_min, box_y_min, box_x_max, box_y_max = box
                box_center = ((box_x_min + box_x_max) / 2, (box_y_min + box_y_max) / 2)

                # Calculate the Euclidean distance between the centers
                center_distance = distance.euclidean(box_center, new_center)

                if center_distance < distance_threshold:
                    # Combine the boxes by expanding the existing box to include the new one
                    bounding_boxes[i] = (
                        min(box_x_min, x),
                        min(box_y_min, y),
                        max(box_x_max, x + w),
                        max(box_y_max, y + h)
                    )
                    added_to_existing = True
                    break

            if not added_to_existing:
                bounding_boxes.append(new_box)

        for box in bounding_boxes:
            x_min, y_min, x_max, y_max = box

            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2

            width = x_max - x_min
            height = y_max - y_min
            # print(f"Top-left: ({x_min}, {y_min}), Bottom-right: ({x_max}, {y_max}), Width: {width}, Height: {height}, Center: ({cx}, {cy})")

        # Threshold for vertical center proximity
        cy_threshold = 3

        # List to store merged bounding boxes
        merged_boxes = []

        # Function to merge boxes based on vertical center proximity
        def merge_boxes_center(bounding_boxes, cy_threshold):
            i = 0
            while i < len(bounding_boxes):
                box = bounding_boxes[i]
                x_min, y_min, x_max, y_max = box
                box_center_y = (y_min + y_max) // 2

                j = i + 1
                while j < len(bounding_boxes):
                    other_box = bounding_boxes[j]
                    other_x_min, other_y_min, other_x_max, other_y_max = other_box
                    other_center_y = (other_y_min + other_y_max) // 2

                    # Check if the vertical centers are within the threshold
                    if abs(box_center_y - other_center_y) <= cy_threshold:
                        # Merge the boxes
                        x_min = min(x_min, other_x_min)
                        y_min = min(y_min, other_y_min)
                        x_max = max(x_max, other_x_max)
                        y_max = max(y_max, other_y_max)

                        # Update the current box
                        bounding_boxes[i] = (x_min, y_min, x_max, y_max)

                        # Remove the other box as it has been merged
                        bounding_boxes.pop(j)
                    else:
                        j += 1

                merged_boxes.append((x_min, y_min, x_max, y_max))
                i += 1

            return merged_boxes

        # Merge the bounding boxes
        merged_boxes = merge_boxes_center(bounding_boxes, cy_threshold)

        # Function to check if two boxes overlap
        def boxes_overlap(box1, box2):
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2

            # Check if one box is to the left of the other or one box is above the other
            if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
                return False
            return True

        # Function to merge two overlapping boxes
        def merge_boxes(box1, box2):
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2

            x_min = min(x1_min, x2_min)
            y_min = min(y1_min, y2_min)
            x_max = max(x1_max, x2_max)
            y_max = max(y1_max, y2_max)

            return x_min, y_min, x_max, y_max

        # Merge overlapping bounding boxes
        def merge_all_boxes(boxes):
            merged = []
            while boxes:
                box = boxes.pop(0)
                merged_box = box
                i = 0
                while i < len(boxes):
                    other_box = boxes[i]
                    if boxes_overlap(merged_box, other_box):
                        # Merge the boxes
                        merged_box = merge_boxes(merged_box, other_box)
                        # Remove the other box
                        boxes.pop(i)
                    else:
                        i += 1
                merged.append(merged_box)
            return merged

        # Merge the bounding boxes
        merged_boxes = merge_all_boxes(bounding_boxes)

        # Print the final merged bounding boxes
        # print("Merged Bounding Boxes:")
        for box in merged_boxes:
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            # print(f"Top-left: ({x_min}, {y_min}), Bottom-right: ({x_max}, {y_max}), Width: {width}, Height: {height}")

        # Print the final merged bounding boxes
        # print("Merged Bounding Boxes:")
        for box in merged_boxes:
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            # print(f"Top-left: ({x_min}, {y_min}), Bottom-right: ({x_max}, {y_max}), Width: {width}, Height: {height}")

        # Print the location and size of all filtered bounding boxes
        # print("Bounding Box Locations and Sizes:")
        for box in merged_boxes:
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            # print(f"Top-left: ({x_min}, {y_min}), Bottom-right: ({x_max}, {y_max}), Width: {width}, Height: {height}")
            if width > 60:
                merged_boxes.remove(box)
        # print('')

        # print("GIGAFINAL Box Locations and Sizes:")
        for box in merged_boxes:
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            if x_min == 0:
                merged_boxes.remove(box)
            # print(f"Top-left: ({x_min}, {y_min}), Bottom-right: ({x_max}, {y_max}), Width: {width}, Height: {height}")

        # Draw the bounding boxes on the image
        for box in merged_boxes:
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)  # Green bounding box

        # Save the result
        # cv2.imwrite('output_with_separate_boxes.png', image)

        # Display the result
        # cv2.imshow('Detected Color Regions with Separate Boxes', image)

        flame_image = cv2.imread('output_image_rgb.png')

        # Display the result
        # cv2.imshow('Detected Flame Stat', flame_image)

        # print(merged_boxes)

        first_two_number = [(x_min, y_min) for x_min, y_min, x_max, y_max in merged_boxes]

        # print(first_two_number)

        # Load the image
        image = cv2.imread('gogogo.png')

        # 1. Scale the Image
        scale_factor = 2
        resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # 2. Apply Less Aggressive Sharpening
        sharpening_kernel = np.array([[0, -1, 0],
                                      [-1, 5, -1],
                                      [0, -1, 0]])

        sharpened = cv2.filter2D(resized_image, -1, sharpening_kernel)

        # 3. Convert the Image to Grayscale
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

        # Save and check the sharpened and grayscale image (optional)
        # cv2.imwrite('debug_resized_sharpened_gray.png', gray)

        original_text = pytesseract.image_to_string(gray, lang='eng')
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

        # 4. Use pytesseract to do OCR on the processed image
        text = original_text.replace('#', '+')

        # print(text)

        # +=5 tolerance
        tolerance = 5

        # To store confirmed section
        confirm_list = []
        confirm_stat = []

        for i in range(len(data['text'])):
            text = data['text'][i]
            if text.strip():  # Only print non-empty text
                x = data['left'][i] / 2
                y = data['top'][i] / 2
                # print(f"{text}, ({x}, {y})")
                for coord in first_two_number:
                    x_ref, y_ref = coord
                    if abs(x - x_ref) <= tolerance and abs(y - y_ref) <= tolerance:
                        confirm_list.append(text)
                        break
                for coord in first_two_number:
                    x_ref, y_ref = coord
                    if x <= 1 and abs(y - y_ref) <= tolerance:
                        confirm_stat.append(text)
                        break
        # print(confirm_list)
        # print(confirm_stat)
        flame_state_string = '=====Flame Stat=====\n'

        combine_flame_stat = list(zip(confirm_stat, confirm_list))

        Str_score = 0
        Dex_score = 0
        Int_score = 0
        Luk_score = 0

        for i in range(len(combine_flame_stat)):
            s1 = f'{combine_flame_stat[i][0]} {combine_flame_stat[i][1]}'
            s2 = s1.replace(')', '')
            s3 = s2.replace('All', 'All Stats')
            cleaned_string = s3.replace('.', '')
            cleaned_string = cleaned_string.replace(':', '')
            flame_state_string += cleaned_string + '\n'

            print(f'j: {combine_flame_stat[i][0]}')

            cleaned_combine_flame_stat = combine_flame_stat[i][0]
            cleaned_combine_flame_stat = cleaned_combine_flame_stat.replace(':', '')

            ### AS applies to all
            if cleaned_combine_flame_stat == 'All':
                stat_string = combine_flame_stat[i][1]
                cleaned_state = stat_string.replace('+', '')
                cleaned_state = cleaned_state.replace('%', '')
                cleaned_state = cleaned_state.replace(')', '')
                cleaned_state = cleaned_state.replace(':', '')
                Str_score += float(cleaned_state) * 10
                Dex_score += float(cleaned_state) * 10
                Int_score += float(cleaned_state) * 10
                Luk_score += float(cleaned_state) * 10

            ### Attack applies to all except INT
            if cleaned_combine_flame_stat == 'Attack':
                stat_string = combine_flame_stat[i][1]
                cleaned_state = stat_string.replace('+', '')
                cleaned_state = cleaned_state.replace('%', '')
                cleaned_state = cleaned_state.replace(')', '')
                cleaned_state = cleaned_state.replace(':', '')
                Str_score += float(cleaned_state) * 3
                Dex_score += float(cleaned_state) * 3
                Luk_score += float(cleaned_state) * 3

            ### Magic applies to INT
            if cleaned_combine_flame_stat == 'Magic':
                stat_string = combine_flame_stat[i][1]
                cleaned_state = stat_string.replace('+', '')
                cleaned_state = cleaned_state.replace('%', '')
                cleaned_state = cleaned_state.replace(')', '')
                cleaned_state = cleaned_state.replace(':', '')
                Int_score += float(cleaned_state) * 3

            ### STR CLASS
            if cleaned_combine_flame_stat == 'STR':
                stat_string = combine_flame_stat[i][1]
                cleaned_state = stat_string.replace('+', '')
                cleaned_state = cleaned_state.replace(')', '')
                cleaned_state = cleaned_state.replace(':', '')
                Str_score += float(cleaned_state)
            if cleaned_combine_flame_stat == 'DEX':
                stat_string = combine_flame_stat[i][1]
                cleaned_state = stat_string.replace('+', '')
                cleaned_state = cleaned_state.replace(')', '')
                cleaned_state = cleaned_state.replace(':', '')
                Str_score += float(cleaned_state) / 10

            ### DEX CLASS
            if cleaned_combine_flame_stat == 'STR':
                stat_string = combine_flame_stat[i][1]
                cleaned_state = stat_string.replace('+', '')
                cleaned_state = cleaned_state.replace(')', '')
                cleaned_state = cleaned_state.replace(':', '')
                Dex_score += float(cleaned_state) / 10
            if cleaned_combine_flame_stat == 'DEX':
                stat_string = combine_flame_stat[i][1]
                cleaned_state = stat_string.replace('+', '')
                cleaned_state = cleaned_state.replace(')', '')
                cleaned_state = cleaned_state.replace(':', '')
                Dex_score += float(cleaned_state)

            ### INT CLASS
            if cleaned_combine_flame_stat == 'INT':
                print('here1')
                stat_string = combine_flame_stat[i][1]
                cleaned_state = stat_string.replace('+', '')
                cleaned_state = cleaned_state.replace(')', '')
                cleaned_state = cleaned_state.replace(':', '')
                Int_score += float(cleaned_state)
            if cleaned_combine_flame_stat == 'LUK':
                print('here')
                stat_string = combine_flame_stat[i][1]
                cleaned_state = stat_string.replace('+', '')
                cleaned_state = cleaned_state.replace(')', '')
                cleaned_state = cleaned_state.replace(':', '')
                Int_score += float(cleaned_state) / 10

            ### LUK CLASS
            if cleaned_combine_flame_stat == 'DEX':
                stat_string = combine_flame_stat[i][1]
                cleaned_state = stat_string.replace('+', '')
                cleaned_state = cleaned_state.replace(')', '')
                cleaned_state = cleaned_state.replace(':', '')
                Luk_score += float(cleaned_state) / 10
            if cleaned_combine_flame_stat == 'LUK':
                stat_string = combine_flame_stat[i][1]
                cleaned_state = stat_string.replace('+', '')
                cleaned_state = cleaned_state.replace(')', '')
                cleaned_state = cleaned_state.replace(':', '')
                Luk_score += float(cleaned_state)

        return f'{flame_state_string}=====Flame Score=====\nSTR Class: {Str_score}\t\tDEX Class: {Dex_score}\nINT Class: {Int_score}\t\tLUK Class: {Luk_score}'

    else:
        # Load the image
        image = cv2.imread(user_image)

        # 1. Scale the Image
        scale_factor = 2.6
        resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # 2. Apply Less Aggressive Sharpening
        sharpening_kernel = np.array([[0, -1, 0],
                                      [-1, 5, -1],
                                      [0, -1, 0]])

        sharpened = cv2.filter2D(resized_image, -1, sharpening_kernel)

        # 3. Convert the Image to Grayscale
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

        # Use Tesseract to get bounding boxes and text
        h, w, _ = resized_image.shape
        boxes = pytesseract.image_to_boxes(gray)  # Gets the bounding boxes

        # Draw boxes and put labels
        for b in boxes.splitlines():
            b = b.split(' ')
            x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            # Adjust y coordinate since OpenCV's origin is top-left, whereas Tesseract's is bottom-left
            y = h - y
            y2 = h - y2

            # Draw the box
            cv2.rectangle(resized_image, (x, y2), (x2, y), (255, 0, 0), 1)

            # Put the recognized text above the box
            cv2.putText(resized_image, b[0], (x, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)

        # Save the image with bounding boxes and labels
        cv2.imwrite('output_with_boxes.png', resized_image)

        detected_text = pytesseract.image_to_string(gray, lang='eng')

        # Perform OCR and get the data with bounding boxes and confidence levels
        data = pytesseract.image_to_data(gray, lang='eng', output_type=pytesseract.Output.DICT)

        # Iterate over each word and print those with a confidence level above 60
        print("Detected Text with Confidence > 60:")
        for i in range(len(data['text'])):
            text = data['text'][i]
            conf = int(data['conf'][i])

            if conf > 10 and text.strip():
                print(f"Text: {text}, Confidence: {conf}")

        #Optional: Show the image
        # cv2.imshow('Detected Text', resized_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return detected_text
