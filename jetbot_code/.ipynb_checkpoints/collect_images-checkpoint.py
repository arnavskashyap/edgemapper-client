import cv2
import time 
from datetime import datetime
import os
from jetbot import Camera, Robot
import glob


def collect_images(image_count, capture_interval, training_data_path, camera, robot, right):
    os.makedirs(training_data_path, exist_ok=True)

    #Clear existing images in training_data_path
    files = glob.glob(os.path.join(training_data_path, "*"))
#     print(training_data_path)
    for f in files:
        os.remove(f)

    image_prefix = 'image'
    image_format = 'jpg'
    image_format_2 = 'png'
#     image_count = image_count
    capture_interval = capture_interval
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if right:
        robot.right(speed=0.25)
    else:
        robot.left(speed=0.25)
    for i in range(image_count):
        # Capture an image from the camera
        image = camera.value
        # Convert the image to the correct format
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Save the image to disk
#         image_path = os.path.join(training_data_path, f'image{i}_{timestamp}.{image_format}')
        image_path = os.path.join(training_data_path, f'{i}.{image_format}')
        image_path_2 = os.path.join(training_data_path, f'{i}.{image_format_2}')
        cv2.imwrite(image_path, image_bgr)
        cv2.imwrite(image_path_2, image_bgr)
        print(f'Captured {image_path}')

        # Pause (if wanted)
        time.sleep(capture_interval)
    robot.stop()
    # Release the camera
    camera.stop()
    print('Image collection complete!')
