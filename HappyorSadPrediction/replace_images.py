import os
import cv2
import shutil

def is_color_image(image_path):
    """Check if an image is in color by verifying if it has three color channels."""
    img = cv2.imread(image_path)
    if img is None:
        return False
    return len(img.shape) == 3 and img.shape[2] == 3  # Check for 3 color channels (BGR)

def replace_images(original_folder, target_folder):
    """Replace images in the target folder with images from the original folder if name matches and the image is in color."""
    for root, _, files in os.walk(original_folder):
        for file in files:
            original_file_path = os.path.join(root, file)
            target_file_path = os.path.join(target_folder, file)

            if os.path.exists(target_file_path):
                # Check if the original image is in color
                if is_color_image(original_file_path):
                    try:
                        shutil.copy2(original_file_path, target_file_path)
                        print(f"Replaced: {target_file_path} with color image from {original_file_path}")
                    except Exception as e:
                        print(f"Error replacing {target_file_path}: {e}")
                else:
                    print(f"Skipped: {original_file_path} (not a color image)")

if __name__ == "__main__":
    # Define folder paths
    original_happy = r"C:\Users\kalop\CV-Project\HappyorSadPrediction\data - Original\happy"
    original_sad = r"C:\Users\kalop\CV-Project\HappyorSadPrediction\data - Original\sad"
    target_happy = r"C:\Users\kalop\CV-Project\HappyorSadPrediction\data\happy"
    target_sad = r"C:\Users\kalop\CV-Project\HappyorSadPrediction\data\sad"

    print("Replacing images in Happy folder...")
    replace_images(original_happy, target_happy)

    print("\nReplacing images in Sad folder...")
    replace_images(original_sad, target_sad)

    print("\nImage replacement process completed.")
