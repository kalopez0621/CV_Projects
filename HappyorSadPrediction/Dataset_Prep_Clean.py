import os

def count_images(folder):
    """Count the number of images in a folder."""
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])

# Define folder paths
happy_folder = r"C:\Users\kalop\CV-Project\HappyorSadPrediction\data\resized_happy"
sad_folder = r"C:\Users\kalop\CV-Project\HappyorSadPrediction\data\resized_sad"

# Count images in each folder
happy_count = count_images(happy_folder)
sad_count = count_images(sad_folder)

print(f"Number of Happy images: {happy_count}")
print(f"Number of Sad images: {sad_count}")

# Check if the dataset is balanced
if happy_count != sad_count:
    print("Warning: The dataset is imbalanced!")
    if happy_count > sad_count:
        print(f"The Happy class has {happy_count - sad_count} more images than the Sad class.")
    else:
        print(f"The Sad class has {sad_count - happy_count} more images than the Happy class.")
else:
    print("The dataset is balanced.")
