# -*- coding: utf-8 -*-
"""


@author: SemihAcmali
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# DeiT modeli için uygun veri dönüşümleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Görüntüyü Tensor formatına çevir
])

# Veri setini yüklüyoruz (veri klasörünün yolu dataset_dir)
dataset_dir = 'D:/DeiT/MangoLeafBDCropped'  # Burada kendi veri seti klasör yolunu kullan

# ImageFolder kullanarak veri setini yükle
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

# Toplam veri seti boyutu
dataset_size = len(dataset)

# Eğitim veri seti için %80 ayırma
train_size = int(0.8 * dataset_size)

# Kalan veri setinin %20'si test ve doğrulama için
remaining_size = dataset_size - train_size

# Kalan veri setini test ve doğrulama olarak %50-%50 bölmek (bu, %10'a %10 yapacak)
val_size = int(0.5 * remaining_size)
test_size = remaining_size - val_size

# random_split ile veri setini üçe böl
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Veri yükleyicileri oluşturma
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

from transformers import DeiTForImageClassification, AutoImageProcessor

# DeiT modelini ve özellik çıkarıcıyı yükle
model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")
# DeiT için özellik çıkarıcı (görüntüleri uygun şekilde işler)
image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")

# Son katmanı, kendi veri setindeki sınıf sayısına göre değiştirmemiz gerekiyor
num_classes = len(dataset.classes)
model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)



# Örnek bir görüntü dönüşüm fonksiyonu
def preprocess(images):
    return image_processor(images=images, return_tensors="pt")

# Optimizatör: sadece sınıflandırıcıyı eğitiyoruz
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.00001)

# Kayıp fonksiyonu
criterion = torch.nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Eğitim döngüsü
epochs = 5  # Eğitim dönemi sayısı
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()  # Modeli eğitim moduna al
    running_loss = 0.0
    
    for images, labels in train_loader:
        # Veriyi GPU'ya taşıma (varsa)
        images, labels = images.to(device), labels.to(device)
        
        # Görüntüleri modelin istediği formata çevir
        inputs = preprocess(images)
        inputs = inputs.to(device)
        
        # Optimizatörü sıfırla
        optimizer.zero_grad()

        # İleri yayılım (forward pass)
        outputs = model(images).logits
        loss = criterion(outputs, labels)

        # Geri yayılım (backward pass) ve ağırlıkları güncelleme
        loss.backward()
        optimizer.step()

        # Kayıpları toplama
        running_loss += loss.item()

    # Eğitim dönemindeki ortalama kaybı hesapla
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}")
    
    # Doğrulama seti ile modelin performansını kontrol etme
    model.eval()  # Modeli doğrulama moduna al
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Doğrulama sırasında gradyan hesaplamayı kapatıyoruz
        for images, labels in val_loader:
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
    
    print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")


# Doğrulama seti ile modelin performansını kontrol etme
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


# 5. Modeli Değerlendirme
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




