import cv2
import time 
import os
import glob
from datetime import datetime
from jetbot import Camera

def collect_images(image_count=50, capture_interval=.2, training_data_path='captured_images'):
    os.makedirs(training_data_path, exist_ok=True)
    
    # Clear existing images in training_data_path
    files = glob.glob(os.path.join(training_data_path, "*"))
    for f in files:
        os.remove(f)
    
    # Initialize camera
    camera = Camera.instance(width=320, height=240)
    image_format = 'jpg'
    
    for i in range(image_count):
        # Capture an image from the camera
        image = camera.value
        
        # Convert the image to the correct format
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Save the image to disk
        image_path = os.path.join(training_data_path, f'{i}.{image_format}')
        cv2.imwrite(image_path, image_bgr)
        print(f'Captured {image_path}')
        
        # Pause before next capture
        time.sleep(capture_interval)
    
    # Release the camera
    camera.stop()
    print('Image collection complete!')

if __name__ == "__main__":
    collect_images()
