# -*- coding: utf-8 -*-
"""


@author: SemihAcmali
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Appropriate data transformations for the DeiT model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Convert the image to Tensor format
])

# Loading the dataset (path to the data folder: dataset_dir)
dataset_dir = 'D:/DeiT/MangoLeafBDCropped'  # Use the path to your own dataset folder here

# Load the dataset using ImageFolder
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

# Total dataset size
dataset_size = len(dataset)

# 80% split for the training dataset
train_size = int(0.8 * dataset_size)

# The remaining 20% of the dataset is reserved for testing and validation
remaining_size = dataset_size - train_size

# Split the remaining dataset 50-50 into test and validation sets (this will result in 10% for each)
val_size = int(0.5 * remaining_size)
test_size = remaining_size - val_size

# Split the dataset into three parts using `random_split`
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Creating data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

from transformers import DeiTForImageClassification, AutoImageProcessor

# Load the DeiT model and feature extractor
model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")
# Feature extractor for DeiT (processes images appropriately)
image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")

# We need to adjust the final layer based on the number of classes in our dataset
num_classes = len(dataset.classes)
model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)



# An example image transformation function
def preprocess(images):
    return image_processor(images=images, return_tensors="pt")

# Optimizer: We are only training the classifier
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.00001)

# Loss Function
criterion = torch.nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Train Loop
epochs = 5  # Number of train loop
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()  
    running_loss = 0.0
    
    for images, labels in train_loader:
        # Transfer data to the GPU (if available)
        images, labels = images.to(device), labels.to(device)
        
        # Convert the images to the format specified by the model
        inputs = preprocess(images)
        inputs = inputs.to(device)
        
        # set the optimizer
        optimizer.zero_grad()

        # (forward pass)
        outputs = model(images).logits
        loss = criterion(outputs, labels)

        # (backward pass) ve weight update
        loss.backward()
        optimizer.step()

        # collect loss
        running_loss += loss.item()

    # Calculate the average loss during the training
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}")
    
    # Evaluating the model's performance using a validation set
    model.eval()  
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # disable gradient calculation during validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
         
            
            # İleri yayılım (forward pass)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            
            # Calculate the validation loss
            val_loss += loss.item()
            
            # (accuracy)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    accuracy = 100 * correct / total
    
    print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")


# Evaluating model performance using a train set(Just test the code)
model.eval()  # Modeli doğrulama moduna al
val_loss = 0.0
correct = 0
total = 0

with torch.no_grad():  # Doğrulama sırasında gradyan hesaplamayı kapatıyoruz
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # inputs = preprocess(images)
        # inputs = inputs.to(device)
        
        # İleri yayılım (forward pass)
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        
        # Doğrulama kaybını hesapla
        val_loss += loss.item()
        
        # Doğruluk (accuracy) hesaplama
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_loss /= len(val_loader)
val_losses.append(val_loss)
accuracy = 100 * correct / total

print(f"Train Loss: {val_loss:.4f}, Train Accuracy: {accuracy:.2f}%")


# 5. Model Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')


#save model
torch.save(model, "D:\\DeiT\\mangoModel\\model.pth")
torch.save(model.state_dict(), "D:/DeiT\\mangoModel\\modelstateDict.pth")

#Section 1

# Load the entire model
model_path = "D:\\DeiT\\mangoModel\\model.pth"
model = torch.load(model_path)
model.to(device)
model.eval()  # Set to evaluation mode

print("Full model loaded successfully.")


#Section 2

from transformers import DeiTForImageClassification

# 1. Re-initialize the architecture
# Ensure num_labels matches the number of classes used during training
model = DeiTForImageClassification.from_pretrained(
    "facebook/deit-base-distilled-patch16-224", 
    num_labels=8  # Replace with your actual num_classes
)

# 2. Load the state dict
state_dict_path = "D:/DeiT/mangoModel/modelstateDict.pth"
state_dict = torch.load(state_dict_path, map_location=device)

# 3. Apply the weights to the model
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("Model state_dict loaded successfully.")
