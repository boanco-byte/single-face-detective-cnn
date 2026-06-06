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

            nn.Linear(128, 5)
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

# ================= PREDICT 1 IMAGE =================
def predict_image(image_path):
    """Predict face detection: returns confidence and bounding box coordinates"""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        # outputs: [confidence, x1, y1, x2, y2]
        confidence = torch.sigmoid(outputs[0, 0]).item()
        
        if confidence > 0.5:
            # Denormalize bounding box from [0, 1] to image dimensions
            x1 = outputs[0, 1].item() * original_size[0]
            y1 = outputs[0, 2].item() * original_size[1]
            x2 = outputs[0, 3].item() * original_size[0]
            y2 = outputs[0, 4].item() * original_size[1]
            
            return {
                "face_detected": True,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
                "width": x2 - x1,
                "height": y2 - y1
            }
        else:
            return {"face_detected": False, "confidence": confidence}


# ================= PREDICT FOLDER =================
def predict_folder(folder_path):
    images = os.listdir(folder_path)

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)

        try:
            result = predict_image(img_path)
            if result["face_detected"]:
                bbox = result["bbox"]
                print(f"{img_name}: FACE DETECTED | Confidence: {result['confidence']:.2f} | BBox: ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})")
            else:
                print(f"{img_name}: No face | Confidence: {result['confidence']:.2f}")
        except Exception as e:
            print(f"Skip: {img_name} (Error: {e})")


# ================= MAIN =================
if __name__ == "__main__":

    # test 1 ảnh
    test_image = "test.jpg"
    print("Single image:", predict_image(test_image))
