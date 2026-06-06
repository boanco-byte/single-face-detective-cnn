import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ================= MODEL =================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.model(x)


# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD CHECKPOINT =================
checkpoint_path = "checkpoints/model.pth"

checkpoint = torch.load(checkpoint_path, map_location=device)

model = CNN().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']} | best acc = {checkpoint['best_acc']:.2f}%")

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ================= CLASS NAMES =================
# chỉnh theo folder của bạn
class_names = ["class_0", "class_1"]

# ================= PREDICT 1 IMAGE =================
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        preds = outputs.argmax(dim=1).item()

    return class_names[preds]


# ================= PREDICT FOLDER =================
def predict_folder(folder_path):
    images = os.listdir(folder_path)

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)

        try:
            pred = predict_image(img_path)
            print(f"{img_name} -> {pred}")
        except:
            print(f"Skip: {img_name}")


# ================= MAIN =================
if __name__ == "__main__":

    # test 1 ảnh
    test_image = "test.jpg"
    print("Single image:", predict_image(test_image))
