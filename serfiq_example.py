# Author: Jan Niklas Kolf, 2020
from face_image_quality import SER_FIQ
import cv2
import os
from pathlib import Path
from tqdm import tqdm

def is_image_ext(file):
    image_extensions = ['.jpg', '.jpeg', '.png','.bmp']
    return any(file.name.lower().endswith(ext) for ext in image_extensions)

def open_dataset(dataset_path):
    input_images = [str(f) for f in sorted(Path(dataset_path).rglob('*')) if is_image_ext(f) and f.is_file()]
    return input_images

if __name__ == "__main__":
    # Sample code of calculating the score of an image
    
    # Create the SER-FIQ Model
    # Choose the GPU, default is 0.
    ser_fiq = SER_FIQ(None)
        
    # Load the test image
    # test_img = cv2.imread("./data/test_img.jpeg")
    
    # # Align the image
    # aligned_img = ser_fiq.apply_mtcnn(test_img)
    
    # # Calculate the quality score of the image
    # # T=100 (default) is a good choice
    # # Alpha and r parameters can be used to scale your
    # # score distribution.
    # score = ser_fiq.get_score(aligned_img, T=100)
    
    # print("SER-FIQ quality score of image 1 is", score)
    
    # # Do the same thing for the second image as well
    # test_img2 = cv2.imread("./data/test_img2.jpeg")
    
    # aligned_img2 = ser_fiq.apply_mtcnn(test_img2)
    
    # score2 = ser_fiq.get_score(aligned_img2, T=100)
   
    # print("SER-FIQ quality score of image 2 is", score2)

    dataset_path = "./testImages/"

    image_files = open_dataset(dataset_path)

    for image_path in tqdm(image_files, desc="Processing images", unit="image"):
        img = cv2.imread(image_path)
        aligned_img = ser_fiq.apply_mtcnn(img)
        if aligned_img is not None:
            score = ser_fiq.get_score(aligned_img, T=100)
            print(score, " ====> ", os.path.basename(image_path))
        else:
            print("Noface ====>" , os.path.basename(image_path))
