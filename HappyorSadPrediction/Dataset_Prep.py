
import os
import cv2

def resize_and_preserve_aspect_ratio(image_path, target_size, output_path):
    # Resize an image to the target size while preserving aspect ratio.
    try:
        img = cv2.imread(image_path)  # Read the image in color mode (default)
        if img is None:
            print(f"Error: Unable to read {image_path}")
            return

        # Get original dimensions
        original_height, original_width = img.shape[:2]
        target_width, target_height = target_size

        # Calculate the scaling factors
        scale_width = target_width / original_width
        scale_height = target_height / original_height
        scale = min(scale_width, scale_height)

        # Compute the new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize the image
        resized_img = cv2.resize(
            img, (new_width, new_height),
            interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        )

        # Add padding to achieve the exact target size
        delta_w = target_width - new_width
        delta_h = target_height - new_height
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        padded_img = cv2.copyMakeBorder(
            resized_img, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]  # Black padding
        )

        # Save the resized image
        cv2.imwrite(output_path, padded_img)
        print(f"Resized and saved: {output_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def resize_images_in_folder(input_folder, output_folder, target_size):
    # Resize all images in the input folder and save them to the output folder.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_folder, file)
            resize_and_preserve_aspect_ratio(input_path, target_size, output_path)

if __name__ == "__main__":
    # Define input and output folders
    input_happy = r"C:\Users\kalop\CV-Project\HappyorSadPrediction\data\happy"
    input_sad = r"C:\Users\kalop\CV-Project\HappyorSadPrediction\data\sad"
    output_happy = r"C:\Users\kalop\CV-Project\HappyorSadPrediction\data\resized_happy"
    output_sad = r"C:\Users\kalop\CV-Project\HappyorSadPrediction\data\resized_sad"

    # Define target size
    target_size = (128, 128)  # Change to your desired resolution

    print("\nResizing Happy images...")
    resize_images_in_folder(input_happy, output_happy, target_size)

    print("\nResizing Sad images...")
    resize_images_in_folder(input_sad, output_sad, target_size)

    print("\nResizing process completed.")


