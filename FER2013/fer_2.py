import os
import random
import numpy as np
import torch
import cv2
from torchvision import transforms, models  # Added models import
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from torch import nn

# Load the fine-tuned model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=None)  # Initialize MobileNetV2
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)  # Set output for 7 classes
model.load_state_dict(torch.load('best_emotion_recognition_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define transformations for test images
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load test dataset with ImageFolder
test_dir = 'FER2013/test'
test_dataset = ImageFolder(test_dir, transform=transform)

# Function to display images in batches with actual and predicted emotions
def display_test_batch(model, test_dataset, batch_size=5, num_batches=4):
    for _ in range(num_batches):
        # Select random indices for the batch
        indices = random.sample(range(len(test_dataset)), batch_size)
        batch = Subset(test_dataset, indices)
        batch_loader = DataLoader(batch, batch_size=batch_size)

        for images, labels in batch_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Get predictions
            with torch.no_grad():
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

            # Display each image with actual and predicted labels
            display_img = np.zeros((400, batch_size * 400, 3), dtype=np.uint8)
            for i in range(batch_size):
                # Convert tensor to image
                img = images[i].cpu().numpy().transpose((1, 2, 0))
                img = ((img * 0.5) + 0.5) * 255  # De-normalize and scale to [0, 255]
                img = cv2.resize(img.astype(np.uint8), (400, 400))

                # Set up actual and predicted text
                actual_text = f"Actual: {emotion_labels[labels[i].item()]}"
                predicted_text = f"Predicted: {emotion_labels[preds[i].item()]}"

                # Add text to the image
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, actual_text, (10, 30), font, 0.7, (0, 0, 255), 2)  # Red for actual
                cv2.putText(img, predicted_text, (10, 60), font, 0.7, (0, 255, 0), 2)  # Green for predicted

                # Add image to display batch
                display_img[0:400, i*400:(i+1)*400] = img

            # Show the batch of images
            cv2.imshow("Emotion Recognition Test Batch", display_img)
            cv2.waitKey(0)  # Press any key to move to the next batch

    cv2.destroyAllWindows()

# Run the test function with random batches
display_test_batch(model, test_dataset, batch_size=5, num_batches=4)
