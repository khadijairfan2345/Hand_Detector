import cv2
import mediapipe as mp
import numpy as np
import os
import natsort
from pathlib import Path
from utils import *
import xml.etree.ElementTree as ET

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # Static mode for individual images
            max_num_hands=2,
            min_detection_confidence=0.1,
            # model_complexity=1
        )

def verify_right_left(model, results, image_width, image_height):
    """
    Verifies if both hands are present (right and left) by comparing pinkie finger landmarks.
    Returns a dictionary indicating the presence of each hand.
    """
    hands_detected = {"right": False, "left": False}
    
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[model.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[model.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      print(
          f'Pinkie finger tip coordinates: (',
          f'{hand_landmarks.landmark[model.mp_hands.HandLandmark.PINKIE_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[model.mp_hands.HandLandmark.PINKIE_FINGER_TIP].y * image_height})'
      )

    return True

def process_image_folder(model, input_folder, output_folder):
    """Process all images in a folder and generate XML annotations."""
    # Supported image extensions
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    # List all image files in the folder
    image_files = [
        f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)
    ]
    
    image_files = natsort.natsorted(image_files, reverse=False)
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process in {input_folder}")
    
    # Create XML root element
    xml_annotations = ET.Element("annotations")
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not read image {image_file}. Skipping.")
            continue
        
        image_name = Path(image_file).stem
        image_id = str(int(image_name.split("_")[-1]))
        image_width, image_height = image.shape[1], image.shape[0]
        image = preprocess_image(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.hands.process(image_rgb)
        hand_count = 0
        seen = set()
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label

                if handedness not in seen:
                    polylines, bounding_box = get_hand_landmarks(
                        hand_landmarks, image_width, image_height
                    )
                    xml_annotation = create_xml_annotation(
                        image_id,
                        image_name,
                        polylines,
                        bounding_box,
                        handedness,
                        image_width,
                        image_height,
                    )
                    xml_annotations.append(xml_annotation)
                    hand_count += 1
                    if hand_count > 2:
                        import pdb

                        pdb.set_trace()
                    print("Found ", hand_count, "hands")
                    if hand_count == 2:
                        verify_right_left(model, results, image_width, image_height)
                    seen.add(handedness)
        else:
            bounding_box = (0, 0, 0, 0)
            polylines = [
                ("thumb", [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]),
                ("index_finger", [(0, 0), (0, 0), (0, 0), (0, 0)]),
                ("middle_finger", [(0, 0), (0, 0), (0, 0), (0, 0)]),
                ("ring_finger", [(0, 0), (0, 0), (0, 0), (0, 0)]),
                ("pinkie_finger", [(0, 0), (0, 0), (0, 0), (0, 0)]),
            ]
            for handdd in ["left", "right"]:
                xml_annotation = create_xml_annotation(
                    image_id,
                    image_name,
                    polylines,
                    bounding_box,
                    handdd,
                    image_width,
                    image_height,
                )
                xml_annotations.append(xml_annotation)
                hand_count += 1
                if hand_count > 2:
                    import pdb

                    pdb.set_trace()
                print("Found ", hand_count, "hands")
                seen.add(handdd)
                
        print(f"Processed {i}/{len(image_files)}: {image_file}")


    output_path = os.path.join(output_folder, input_folder.split("/")[-1] + ".xml")
    tree = ET.ElementTree(xml_annotations)
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    # Write XML file
    with open(output_path, "wb") as f:  # Open in binary write mode
        tree.write(f, encoding="utf-8", xml_declaration=True)
    print(f"/nAll images processed successfully! Annotations saved to: {output_path}")


# Example usage
model = HandDetector()
folders = os.listdir("./")
input_folder = "D:/osama__1/frames"
output_folder = "./output_folder"
process_image_folder(model, input_folder, output_folder)
model.hands.close()
