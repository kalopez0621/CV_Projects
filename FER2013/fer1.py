from tensorflow.keras.models import load_model
import numpy as np
import cv2  # Use OpenCV for image preprocessing and display

# Load the saved model
model = load_model('final_emotion_recognition_model.keras')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to preprocess a single image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image file not found or could not be read: {image_path}")
    img = cv2.resize(img, (48, 48))
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale
    return img

# Function to predict emotion and display the image with label
def predict_and_display_emotion(image_path):
    # Preprocess the image
    img_for_model = preprocess_image(image_path)
    # Make prediction
    prediction = model.predict(img_for_model)
    emotion_class = np.argmax(prediction)  # Get the predicted class index
    predicted_emotion = emotion_labels[emotion_class]
    
    # Load the original image in color for display
    img_to_display = cv2.imread(image_path)
    # Resize window to fit the image better
    img_to_display = cv2.resize(img_to_display, (400, 400))

    # Set the label position to bottom left corner
    label_position = (10, img_to_display.shape[0] - 10)  # 10 pixels above the bottom left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)  # Green color for the label text
    thickness = 2

    # Add label to the image
    cv2.putText(img_to_display, f"Emotion: {predicted_emotion}", label_position, font, font_scale, color, thickness)

    # Show the image with the emotion label
    cv2.imshow("Emotion Recognition", img_to_display)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close the display window

    # Print the predicted emotion in the console as well
    print(f"Predicted Emotion: {predicted_emotion}")

# Test the model with a sample image (replace with the path to your test image)
image_path = 'C:/Users/kalop/CV-Project/FER2013/test/angry/PrivateTest_88305.jpg'  # Updated path
predict_and_display_emotion(image_path)
