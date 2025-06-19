import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

def preprocess_fingerprint_image(input_image_path, output_image_path):
    """
    Preprocess a fingerprint image by applying Gaussian and median blurs, then skeletonizing the image.
    
    Parameters:
    input_image_path (str): The path to the input fingerprint image.
    output_image_path (str): The path to save the preprocessed fingerprint image.
    """
    # Load the input image
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur
    gauss_blur = cv2.GaussianBlur(image, (1, 1), 0)
    
    # Apply median blur
    median_blur = cv2.medianBlur(gauss_blur, 1)
    
    # Convert image to binary
    _, binary_image = cv2.threshold(median_blur, 127, 255, cv2.THRESH_BINARY)
    
    # Skeletonize the image
    # skeleton = skeletonize(binary_image // 255) * 255
    # skeleton = skeleton.astype(np.uint8)
    
    # Save the preprocessed image
    cv2.imwrite(output_image_path, binary_image)
    
    # Optional: Display the original and preprocessed images
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Preprocessed Image')
    #plt.imshow(skeleton, cmap='gray')
    plt.show()

# Example usage:
# preprocess_fingerprint_image('path_to_input_image.bmp', 'path_to_output_image.bmp')

#preprocess_fingerprint_image('DATABASE\left_thumb.jpg','new\left_thumb_pp.jpg')