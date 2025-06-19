import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# Load the pre-trained ResNet50 model
half_mark_classifier = keras.models.load_model("resNet_model_for_fingerprintClassification_hdf5_96%.h5")

# Function to classify the fingerprint
def classify_fingerprint(img_path):
    test_image = image.load_img(img_path, target_size=(256, 256))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    predictions = half_mark_classifier.predict(test_image)

    class_names = ["arch", "left_loop", "right_loop", "tented_arch", "whirl"]
    prediction_dict = {class_names[i]: float(predictions[0][i]) * 100 for i in range(len(class_names))}

    # Print classification probabilities
    print("\nFingerprint Classification Probabilities:")
    for class_name, prob in prediction_dict.items():
        print(f"{class_name}: {prob:.2f}%")

    sorted_classes = sorted(prediction_dict, key=prediction_dict.get, reverse=True)
    return sorted_classes


# Load image paths from a directory
def load_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Bidirectional batch search function
def bidirectional_batch_search(image_paths, batch_size, input_des, kp1):
    sift = cv.SIFT_create()
    total_files = len(image_paths)
    best_match_info = {
        "file": None,
        "matches": [],
        "percentage": 0,
        "image": None
    }

    start = 0
    end = total_files - 1

    # Use tqdm for a progress bar
    progress_bar = tqdm(total=total_files, desc="Searching files", unit="file")

    while start <= end:
        # Process batch from the start
        for i in range(start, min(start + batch_size, total_files)):
            if process_file(image_paths[i], input_des, kp1, sift, best_match_info):
                progress_bar.update(total_files - progress_bar.n)  # Complete the progress bar
                progress_bar.close()
                return best_match_info  # Early termination if a perfect match is found
            progress_bar.update(1)

        start += batch_size

        # Process batch from the end
        for i in range(max(end - batch_size + 1, 0), end + 1):
            if process_file(image_paths[i], input_des, kp1, sift, best_match_info):
                progress_bar.update(total_files - progress_bar.n)  # Complete the progress bar
                progress_bar.close()
                return best_match_info  # Early termination if a perfect match is found
            progress_bar.update(1)

        end -= batch_size

    progress_bar.close()
    return best_match_info

# Helper function to process individual files
def process_file(file_path, input_des, kp1, sift, best_match_info):
    frame = cv.imread(file_path)
    kp2, des2 = sift.detectAndCompute(frame, None)

    if des2 is None:  # Handle cases where no keypoints are detected
        return False

    flann = cv.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
    matches = flann.knnMatch(np.asarray(input_des, np.float32), np.asarray(des2, np.float32), k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    match_percentage = (len(good_matches) / len(kp1)) * 100

    if match_percentage > best_match_info["percentage"]:
        best_match_info.update({
        "file": file_path,
        "matches": good_matches,  # Save the actual match objects
        "percentage": match_percentage,
        "image": frame
})


    # Early termination if perfect match is found
    return match_percentage == 100

# Main function to find the best match for an input fingerprint
def match_fingerprint(input_img_path, base_search_path):
    sift = cv.SIFT_create()

    input_img = cv.imread(input_img_path)
    input_img = input_img.astype('uint8')
    kp1, des1 = sift.detectAndCompute(input_img, None)

    sorted_classes = classify_fingerprint(input_img_path)

    best_match_info = {
        "file": None,
        "matches": 0,
        "percentage": 0,
        "image": None
    }

    for class_name in sorted_classes:
        search_path = os.path.join(base_search_path, class_name.replace(" ", "_").lower())
        print(f"\nSearching in folder: {search_path}")
        image_paths = load_image_paths(search_path)
        
        # Perform bidirectional batch search
        best_match_info = bidirectional_batch_search(image_paths, batch_size=10, input_des=des1, kp1=kp1)

        if best_match_info["percentage"] == 100:
            break

    if best_match_info["percentage"] > 0:
        print(f"Best Match: {best_match_info['file']} ({best_match_info['percentage']:.2f}%)")
        
        # Optional: Visualize the matches
        draw_params = dict(
            matchColor=(0, 255, 0),  # Green matches
            singlePointColor=None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        img_matches = cv.drawMatches(
            input_img, kp1, best_match_info["image"], 
            sift.detectAndCompute(best_match_info["image"], None)[0], 
            [m for m in best_match_info["matches"]], None, **draw_params
        )
        cv.imshow("Best Match", img_matches)
        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("No match found.")

    return best_match_info

# Example usage
input_img_path = r"{Input Image Path}"
base_search_path = r"{Searching Folder Path}"

match_info = match_fingerprint(input_img_path, base_search_path)
