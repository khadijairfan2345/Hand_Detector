import cv2
import mediapipe as mp
import numpy as np
import xml.etree.ElementTree as ET
import os
import natsort
from pathlib import Path


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
    

def preprocess_image(image):
    """Preprocess the input image for better detection."""
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.convertScaleAbs(image, alpha=1.3, beta=15)
    return image

def get_hand_landmarks(hand_landmarks, image_width, image_height):
    """Extract hand landmarks and bounding box."""
    landmark_points = {}
    for idx, landmark in enumerate(hand_landmarks.landmark):
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)
        landmark_points[idx] = (x, y)
    polylines = []
    # Thumb: 5 points (0,1,2,3,4)
    thumb_points = [landmark_points[i] for i in range(5)]
    polylines.append(("thumb", thumb_points))
    # Fingers: Index, Middle, Ring, Pinky (4 points each)
    finger_indices = [(5, 9), (9, 13), (13, 17), (17, 21)]
    finger_names = ["index_finger", "middle_finger", "ring_finger", "pinkie_finger"]
    for name, (start, end) in zip(finger_names, finger_indices):
        points = [landmark_points[i] for i in range(start, end)]
        polylines.append((name, points))
    # Calculate bounding box
    x_coords = [x for x, y in landmark_points.values()]
    y_coords = [y for x, y in landmark_points.values()]
    x_min = max(0, min(x_coords) - 6)
    x_max = min(image_width, max(x_coords) + 6)
    y_min = max(0, min(y_coords) - 6)
    y_max = min(image_height, max(y_coords) + 6)
    return polylines, (x_min, y_min, x_max, y_max)




def create_xml_annotation(
    image_id, image_name, polylines, bounding_box, handedness, image_width, image_height
):
    """Create XML annotation for a single image."""
    xml_handedness = "right" if handedness == "Left" else "left"
    image_element = ET.Element(
        "image",
        id=image_id,
        name=image_name,
        width=str(image_width),
        height=str(image_height),
    )
    # Add bounding box
    hand_element = ET.SubElement(
        image_element,
        "box",
        label="hand",
        xtl=str(bounding_box[0]),
        ytl=str(bounding_box[1]),
        xbr=str(bounding_box[2]),
        ybr=str(bounding_box[3]),
        occluded="0",
    )
    ET.SubElement(hand_element, "attribute", name="hand_type").text = xml_handedness
    # Add polylines
    for finger, points in polylines:
        points_str = ";".join(f"{x},{y}" for x, y in points)
        polyline_element = ET.SubElement(
            image_element, "polyline", label=finger, points=points_str, occluded="0"
        )
        ET.SubElement(
            polyline_element, "attribute", name="hand_type"
        ).text = xml_handedness
    return image_element


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
    # os.makedirs(output_folder, exist_ok=True)
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
    # Save XML file
    # Save XML file
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
