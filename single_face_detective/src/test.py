import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ================= MODEL =================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 512),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((3, 3)),

            nn.Flatten(),

            nn.Linear(in_features = 512 * 3 * 3, out_features = 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(in_features = 128, out_features = 5)
        )

    def forward(self, x):
        return self.model(x)


# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD CHECKPOINT =================
current_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(current_dir, '..', 'checkpoints/model.pth')

checkpoint = torch.load(checkpoint_path, map_location=device)

model = CNN().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

#print(f"Loaded model from epoch {checkpoint['epoch']} | best acc = {checkpoint['best_acc']:.2f}%")

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
        # outputs: [confidence, cx, cy, w, h]
        confidence = torch.sigmoid(outputs[0, 0]).item()
        
        if confidence <= 0.5:
            # Convert predicted YOLO-format center box to image coordinates
            bbox = torch.sigmoid(outputs[0, 1:5])
            cx = bbox[0].item() * original_size[0]
            cy = bbox[1].item() * original_size[1]
            w = bbox[2].item() * original_size[0]
            h = bbox[3].item() * original_size[1]

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # Clamp coordinates to image bounds
            x1 = max(0, min(x1, original_size[0]))
            y1 = max(0, min(y1, original_size[1]))
            x2 = max(0, min(x2, original_size[0]))
            y2 = max(0, min(y2, original_size[1]))
            
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
# ================= MAIN =================
if __name__ == "__main__":
    test_image_path = r"C:/CNN/dataset/train/images/ashton-video_mov-81_jpg.rf.EqYEUZpxO6yVYNWbx5hY.jpg"
    print("1. Đang đọc ảnh bằng OpenCV...")
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print(f"Không thể mở hoặc tìm thấy ảnh tại đường dẫn: {test_image_path}")
        exit()
    
    print("2. Đang đưa ảnh vào hàm predict_image...")
    result = predict_image(test_image_path)
    
    print("3. Kết quả hàm predict trả về là:", result) # Dòng này cực kỳ quan trọng để check lỗi ẩn
    
    if result.get("face_detected", False):
        x1, y1, x2, y2 = map(int, result['bbox'])
        out_image = cv2.rectangle(test_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
       
        face_prob = (1.0 - result['confidence']) * 100
        label = f"Face: {face_prob:.1f}%"
        
        cv2.putText(out_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 
        cv2.imshow("Result", out_image)
    else:
        cv2.imshow("Result", test_image)
        
    print("4. Đang đợi bấm phím để đóng cửa sổ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("5. Chương trình kết thúc an toàn!")