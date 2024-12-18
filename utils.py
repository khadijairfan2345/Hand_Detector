import xml.etree.ElementTree as ET
import cv2


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