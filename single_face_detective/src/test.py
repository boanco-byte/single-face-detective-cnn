import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# ================= MODEL =================
class FaceDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU(),

            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),

            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(),

            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),

            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),
        )


        self.conf_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # Bounding box head
        self.boxes_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid()  # Keep bbox in [0, 1]
        )
    
    def forward(self, x):
       backbone = self.backbone(x)
       conf = self.conf_head(backbone)
       boxes = self.boxes_head(backbone)

       return conf, boxes


# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(current_dir, '..', 'checkpoints/model.pth')

checkpoint = torch.load(checkpoint_path, map_location=device)

backbone = FaceDetector().to(device)
backbone.load_state_dict(checkpoint["model_state_dict"])
backbone.eval()

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ================= PREDICT 1 IMAGE =================
def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_conf, pred_boxes = backbone(image) 
        pred_conf = pred_conf.squeeze(1)
        
        if pred_conf <= 0.5:

            pred_cx, pred_cy, pred_w, pred_h = pred_boxes[:,0], pred_boxes[:,1], pred_boxes[:,2], pred_boxes[:,3]

            pred_x1 = pred_cx - pred_w / 2
            pred_y1 = pred_cy - pred_h / 2
            pred_x2 = pred_cx + pred_w / 2
            pred_y2 = pred_cy + pred_h / 2

            x1 = pred_x1 * original_size[0]
            y1 = pred_y1 * original_size[1]
            x2 = pred_x2 * original_size[0]
            y2 = pred_y2 * original_size[1]

            # Clamp coordinates to image bounds
            x1 = int(max(0, min(x1, original_size[0])))
            y1 = int(max(0, min(y1, original_size[1])))
            x2 = int(max(0, min(x2, original_size[0])))
            y2 = int(max(0, min(y2, original_size[1])))
            
            return {
                "face_detected": True,
                "confidence": pred_conf,
                "bbox": [x1, y1, x2, y2],
                "width": x2 - x1,
                "height": y2 - y1
            }
        else:
            return {"face_detected": False, "confidence": pred_conf}


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
    test_image_path = r"C:/Python/single_face_detective/dataset/train/images/IMG_8083_jpg.rf.PcxI2kfPRio7KMfOgtKQ.jpg"
    print("1. Đang đọc ảnh bằng OpenCV...")
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print(f"Không thể mở hoặc tìm thấy ảnh tại đường dẫn: {test_image_path}")
        exit()
    

    print("2. Đang đưa ảnh vào hàm predict_image...")
    result = predict_image(test_image_path)
    
    print("3. Kết quả hàm predict trả về là:", result) # Dòng này cực kỳ quan trọng để check lỗi ẩn
    
    if result.get("face_detected", True):
        x1, y1, x2, y2 = map(int, result['bbox'])
        

        out_image = cv2.rectangle(test_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Result", out_image)
    else:
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Result", test_image)
        
    print("4. Đang đợi bấm phím để đóng cửa sổ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("5. Chương trình kết thúc an toàn!")