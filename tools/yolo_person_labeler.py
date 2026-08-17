"""
Automatic Person Labeler for YOLO
Detects persons automatically and generates annotations in YOLO format

Controls:
- SPACE: Process current image automatically
- A: Process all images automatically
- N: Next image
- P: Previous image
- S: Save current annotations
- R: Review detections (show/hide)
- Q: Quit
"""

import cv2
import numpy as np
import os
import glob
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

class YOLOPersonLabeler:
    def __init__(self, images_folder=None, labels_folder=None):
        self.root = tk.Tk()
        self.root.withdraw()

        # Select folders if not specified
        if images_folder is None:
            self.images_folder = self.select_images_folder()
        else:
            self.images_folder = images_folder
            
        if labels_folder is None:
            self.labels_folder = self.select_labels_folder()
        else:
            self.labels_folder = labels_folder
            
        if not self.images_folder or not self.labels_folder:
            print("Error: required folders were not selected")
            return
            
        self.current_image_index = 0
        self.image_files = []
        self.current_image = None
        self.current_image_path = ""
        self.display_image = None
        
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_boxes = []

        self.window_name = "YOLO Auto Person Labeler"
        self.box_color = (0, 255, 0)        # green
        self.text_color = (255, 255, 255)   # white
        self.detection_color = (0, 0, 255)  # red for automatic detections

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.show_detections = True
        self.auto_detections = []
        self.processing_all = False

        os.makedirs(self.labels_folder, exist_ok=True)

        self.load_image_list()

        print("=== YOLO Person Labeler ===")
        print(f"Images folder : {self.images_folder}")
        print(f"Labels folder : {self.labels_folder}")
        print(f"Images found  : {len(self.image_files)}")
        
    def load_image_list(self):
        """Load image file list."""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.image_files = []
        
        for ext in extensions:
            pattern = os.path.join(self.images_folder, ext)
            self.image_files.extend(glob.glob(pattern))
            pattern_upper = os.path.join(self.images_folder, ext.upper())
            self.image_files.extend(glob.glob(pattern_upper))
        
        self.image_files.sort()
        
    def select_images_folder(self):
        """Select images folder via GUI dialog."""
        messagebox.showinfo("Folder selection", "Select the folder containing the images to label")
        folder = filedialog.askdirectory(
            title="Select images folder",
            initialdir=os.getcwd()
        )
        return folder if folder else None

    def select_labels_folder(self):
        """Select destination folder for YOLO labels."""
        messagebox.showinfo("Folder selection", "Select the folder where YOLO labels will be saved")
        folder = filedialog.askdirectory(
            title="Select labels folder",
            initialdir=os.getcwd()
        )
        return folder if folder else None
        
    def detect_persons_auto(self):
        """Detect persons automatically using HOG."""
        if self.current_image is None:
            return []

        print(f"Auto-detecting persons in: {os.path.basename(self.current_image_path)}")

        rectangles, weights = self.hog.detectMultiScale(
            self.current_image,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.05,
            hitThreshold=0.5,
            groupThreshold=2
        )

        valid_detections = []
        min_area = 1500       # minimum area
        min_confidence = 0.8

        for i, (x, y, w, h) in enumerate(rectangles):
            if i < len(weights):
                confidence = weights[i]
                area = w * h
                aspect_ratio = h / w if w > 0 else 0

                if (confidence > min_confidence and
                        area > min_area and
                        1.2 < aspect_ratio < 4.0):  # typical person aspect ratio
                    valid_detections.append(((x, y), (x + w, y + h)))

        print(f"Persons detected: {len(valid_detections)}")
        return valid_detections
    
    def process_current_image_auto(self):
        """Process current image automatically."""
        self.auto_detections = self.detect_persons_auto()
        self.current_boxes = self.auto_detections.copy()

        if self.current_boxes:
            self.save_annotations()
        
        self.update_display()
        return len(self.current_boxes)
    
    def process_all_images_auto(self):
        """Process all images automatically."""
        if not self.image_files:
            print("No images to process")
            return

        self.processing_all = True
        total_processed = 0
        total_persons = 0

        print(f"\n=== AUTOMATIC PROCESSING STARTED ===")
        print(f"Processing {len(self.image_files)} images...")
        print("-" * 50)

        for i, image_path in enumerate(self.image_files):
            self.current_image_index = i

            if not self.load_current_image():
                continue

            persons_detected = self.process_current_image_auto()

            if persons_detected > 0:
                total_processed += 1
                total_persons += persons_detected

            filename = os.path.basename(image_path)
            print(f"[{i+1}/{len(self.image_files)}] {filename}: {persons_detected} persons")

            if i % 5 == 0:
                self.update_display()
                cv2.waitKey(1)

        self.processing_all = False

        print("-" * 50)
        print(f"=== PROCESSING COMPLETE ===")
        print(f"Images processed : {total_processed}/{len(self.image_files)}")
        print(f"Total persons    : {total_persons}")
        print(f"Avg per image    : {total_persons/len(self.image_files):.2f}")

        self.current_image_index = 0
        self.load_current_image()
        
    def load_current_image(self):
        """Load current image and its existing annotations."""
        if not self.image_files:
            print("No images to process")
            return False

        self.current_image_path = self.image_files[self.current_image_index]
        self.current_image = cv2.imread(self.current_image_path)

        if self.current_image is None:
            print(f"Error loading image: {self.current_image_path}")
            return False

        self.display_image = self.current_image.copy()
        self.load_existing_annotations()
        self.update_display()
        return True
    
    def load_existing_annotations(self):
        """Load existing annotations if present."""
        self.current_boxes = []

        image_name = Path(self.current_image_path).stem
        label_path = os.path.join(self.labels_folder, f"{image_name}.txt")

        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                h, w = self.current_image.shape[:2]

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id == 0:  # person class (class 0 in YOLO)
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])

                            # Convert from YOLO format to pixel coordinates
                            x1 = int((x_center - width/2) * w)
                            y1 = int((y_center - height/2) * h)
                            x2 = int((x_center + width/2) * w)
                            y2 = int((y_center + height/2) * h)

                            self.current_boxes.append(((x1, y1), (x2, y2)))

            except Exception as e:
                print(f"Error loading annotations: {e}")
    
    def save_annotations(self):
        """Save annotations in YOLO format."""
        if not self.current_image_path or not self.current_boxes:
            return

        image_name = Path(self.current_image_path).stem
        label_path = os.path.join(self.labels_folder, f"{image_name}.txt")

        h, w = self.current_image.shape[:2]

        try:
            with open(label_path, 'w') as f:
                for start_point, end_point in self.current_boxes:
                    x1, y1 = start_point
                    x2, y2 = end_point

                    # Ensure coordinates are in the correct order
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)

                    # Convert to normalised YOLO format
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h

                    # class 0 = person
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\\n")

            print(f"Annotations saved: {label_path} ({len(self.current_boxes)} persons)")

        except Exception as e:
            print(f"Error saving annotations: {e}")
    
    def update_display(self):
        """Refresh the displayed image with bounding boxes."""
        self.display_image = self.current_image.copy()

        if self.show_detections:
            for i, (start_point, end_point) in enumerate(self.current_boxes):
                color = self.detection_color if hasattr(self, 'auto_detections') else self.box_color
                cv2.rectangle(self.display_image, start_point, end_point, color, 2)

                label = f"Person {i+1}"
                label_pos = (start_point[0], start_point[1] - 10)
                cv2.putText(self.display_image, label, label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

        info_text = f"Image: {self.current_image_index + 1}/{len(self.image_files)} | Persons: {len(self.current_boxes)}"
        cv2.putText(self.display_image, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        filename = os.path.basename(self.current_image_path)
        cv2.putText(self.display_image, filename, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if self.processing_all:
            cv2.putText(self.display_image, "PROCESSING AUTOMATICALLY...", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if hasattr(self, 'auto_detections') and self.auto_detections:
            cv2.putText(self.display_image, "AUTO-DETECTED", (10, self.display_image.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow(self.window_name, self.display_image)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse event callback."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                self.update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.end_point = (x, y)

                # Require minimum box size
                if (abs(self.end_point[0] - self.start_point[0]) > 10 and
                        abs(self.end_point[1] - self.start_point[1]) > 10):
                    self.current_boxes.append((self.start_point, self.end_point))
                    print(f"Person added: {len(self.current_boxes)}")

                self.start_point = None
                self.end_point = None
                self.update_display()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Cancel current box or remove the last one
            if self.drawing:
                self.drawing = False
                self.start_point = None
                self.end_point = None
                self.update_display()
            elif self.current_boxes:
                self.current_boxes.pop()
                print(f"Person removed. Remaining: {len(self.current_boxes)}")
                self.update_display()
    
    def next_image(self):
        """Go to the next image."""
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
        else:
            print("This is the last image")

    def previous_image(self):
        """Go to the previous image."""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
        else:
            print("This is the first image")
    
    def print_instructions(self):
        """Print keyboard controls."""
        print("\n=== CONTROLS — AUTOMATIC PERSON LABELER ===")
        print("SPACE : Auto-detect persons in current image")
        print("A     : Process ALL images automatically")
        print("N     : Next image")
        print("P     : Previous image")
        print("S     : Save current annotations")
        print("R     : Show/hide detections")
        print("D     : Delete all annotations for current image")
        print("H     : Show these instructions")
        print("Q     : Quit")
        print("============================================\n")
    
    def run(self):
        """Run the application."""
        if not hasattr(self, 'images_folder') or not self.images_folder:
            print("Error: no images folder selected")
            return

        if not self.image_files:
            print(f"No images found in: {self.images_folder}")
            print("Supported formats: JPG, JPEG, PNG, BMP")
            return

        self.print_instructions()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        if not self.load_current_image():
            return

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:     # Q or Esc
                break
            elif key == ord(' '):                 # Space — auto-detect
                self.process_current_image_auto()
            elif key == ord('a'):                 # A — process all
                self.process_all_images_auto()
            elif key == ord('s'):                 # S — save
                self.save_annotations()
            elif key == ord('n'):                 # N — next
                self.next_image()
            elif key == ord('p'):                 # P — previous
                self.previous_image()
            elif key == ord('r'):                 # R — show/hide detections
                self.show_detections = not self.show_detections
                self.update_display()
                print(f"Detections {'shown' if self.show_detections else 'hidden'}")
            elif key == ord('d'):                 # D — delete all annotations
                self.current_boxes = []
                self.auto_detections = []
                print("All annotations deleted")
                self.update_display()
            elif key == ord('h'):                 # H — help
                self.print_instructions()

        cv2.destroyAllWindows()
        print("Labeling complete")

def main():
    print("Initialising YOLO Person Labeler...")
    print("Folder selection dialogs will open...")

    labeler = YOLOPersonLabeler()

    if hasattr(labeler, 'images_folder') and labeler.images_folder:
        labeler.run()
    else:
        print("Could not initialise the labeler. Exiting.")

if __name__ == "__main__":
    main()