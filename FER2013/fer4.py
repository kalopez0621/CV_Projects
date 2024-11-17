import os
import random
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the fine-tuned model
model = load_model('final_emotion_recognition_model_finetuned.keras')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Path to the test dataset
test_dir = 'FER2013/test'

# Function to preprocess a single image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to get 20 random test images and their labels
def get_random_test_images(test_dir, num_images=20):
    class_folders = os.listdir(test_dir)
    random_images = []
    
    for _ in range(num_images):
        # Randomly select an emotion folder
        emotion_folder = random.choice(class_folders)
        emotion_folder_path = os.path.join(test_dir, emotion_folder)
        
        # Randomly select an image from this folder
        image_name = random.choice(os.listdir(emotion_folder_path))
        image_path = os.path.join(emotion_folder_path, image_name)
        
        # Append the path and actual emotion label
        random_images.append((image_path, emotion_folder))
    
    return random_images

# Function to display images in batches of 5 with actual and predicted emotions
def display_test_images_with_predictions_in_batches(model, test_dir, batch_size=5):
    # Get 20 random images with their actual emotions
    random_images = get_random_test_images(test_dir, num_images=20)
    
    for i in range(0, len(random_images), batch_size):
        batch = random_images[i:i+batch_size]
        
        # Create a blank image to display the batch
        display_img = np.zeros((400, batch_size * 400, 3), dtype=np.uint8)
        
        for j, (image_path, actual_emotion) in enumerate(batch):
            # Preprocess the image
            img_for_model = preprocess_image(image_path)
            
            # Predict the emotion
            prediction = model.predict(img_for_model)
            predicted_emotion = emotion_labels[np.argmax(prediction)]
            
            # Load the image for display
            img_to_display = cv2.imread(image_path)
            img_to_display = cv2.resize(img_to_display, (400, 400))
            
            # Set up display text for actual and predicted emotions
            actual_text = f"Actual: {actual_emotion}"
            predicted_text = f"Predicted: {predicted_emotion}"
            
            # Set positions for the texts
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            color_actual = (0, 0, 255)  # Red color for actual label
            color_predicted = (0, 255, 0)  # Green color for predicted label
            thickness = 2
            
            # Add actual and predicted emotion text on the image
            cv2.putText(img_to_display, actual_text, (10, 30), font, font_scale, color_actual, thickness)
            cv2.putText(img_to_display, predicted_text, (10, 60), font, font_scale, color_predicted, thickness)
            
            # Place the image in the batch display
            display_img[0:400, j*400:(j+1)*400] = img_to_display
        
        # Show the batch of 5 images
        cv2.imshow("Emotion Recognition Test Batch", display_img)
        cv2.waitKey(0)  # Wait until a key is pressed to move to the next batch
    
    cv2.destroyAllWindows()  # Close the display window

# Run the test function
display_test_images_with_predictions_in_batches(model, test_dir)
