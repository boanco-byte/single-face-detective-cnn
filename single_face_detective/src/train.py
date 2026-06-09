import os
import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.ops as ops
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

LEARNING_RATE = 3e-4
EPOCHS = 20
LAMBDA_CONF = 1.0
LAMBDA_BOX = 5.0

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ================= TRANSFORMS =================
train_transforms = transforms.Compose([
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
    transforms.RandomGrayscale( p= 0.1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

def target_transforms(string_data):
    if not string_data.strip():
        return torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    
    elements = [float(x) for x in string_data.split()]
    cx, cy, w, h = elements[1:5]

    return torch.tensor([0.0, cx, cy, w, h])

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
    
# ================= DATA =================
class FaceData(Dataset):
    def __init__(self, img_dir, label_dir, transform = None, target_transform = None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_transform = target_transform

        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        all_label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        label_map = {os.path.splitext(f)[0]: f for f in all_label_files}

        self.label_files = []
        missing_labels = []
        for img_file in self.img_files:
            base_name = os.path.splitext(img_file)[0]
            if base_name in label_map:
                self.label_files.append(label_map[base_name])
            else:
                missing_labels.append(img_file)

        if missing_labels:
            raise ValueError(
                f"Không tìm thấy nhãn cho {len(missing_labels)} ảnh. Ví dụ: {missing_labels[:5]}"
            )

        if len(self.img_files) != len(self.label_files):
            raise ValueError(
                f"Số lượng ảnh ({len(self.img_files)}) và nhãn ({len(self.label_files)}) không khớp nhau!"
            )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        label_name = self.label_files[idx]
        label_path = os.path.join(self.label_dir, label_name)

        with open(label_path, 'r') as f:
            string_data = f.readline().strip()

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(string_data)

        return img, label

current_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(current_dir, '..', 'dataset', 'train')
VAL_DIR = os.path.join(current_dir, '..', 'dataset', 'valid')

if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(f"Train directory not found: {TRAIN_DIR}")

if not os.path.isdir(VAL_DIR):
    raise FileNotFoundError("Validation folder missing. Do not use test as validation.")

train_dataset = FaceData(
    img_dir = os.path.join(TRAIN_DIR, 'images'),
    label_dir = os.path.join(TRAIN_DIR, 'labels'),
    transform = train_transforms,
    target_transform = target_transforms
)

val_dataset = FaceData(
    img_dir = os.path.join(VAL_DIR, 'images'),
    label_dir = os.path.join(VAL_DIR, 'labels'),
    transform = val_transforms,
    target_transform = target_transforms
)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang chạy cấu hình trên thiết bị: {device}")
model = CNN()
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), LEARNING_RATE)

best_iou = 0.0
for epoch in range(EPOCHS):

    # ================= TRAIN =================
    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images) 

        pred_conf = outputs[:, 0]
        pred_boxes = torch.sigmoid(outputs[:,1:5])
        true_conf = labels[:, 0]
        true_boxes = labels[:, 1:5]

        # Convert YOLO-format boxes (cx, cy, w, h) to xyxy for IoU/loss
        pred_cx, pred_cy, pred_w, pred_h = pred_boxes[:,0], pred_boxes[:,1], pred_boxes[:,2], pred_boxes[:,3]
        true_cx, true_cy, true_w, true_h = true_boxes[:,0], true_boxes[:,1], true_boxes[:,2], true_boxes[:,3]

        pred_x1 = pred_cx - pred_w / 2
        pred_y1 = pred_cy - pred_h / 2
        pred_x2 = pred_cx + pred_w / 2
        pred_y2 = pred_cy + pred_h / 2
        pred_boxes_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1).clamp(0.0, 1.0)

        true_x1 = true_cx - true_w / 2
        true_y1 = true_cy - true_h / 2
        true_x2 = true_cx + true_w / 2
        true_y2 = true_cy + true_h / 2
        true_boxes_xyxy = torch.stack([true_x1, true_y1, true_x2, true_y2], dim=1).clamp(0.0, 1.0)

        loss_conf = criterion(pred_conf, true_conf)

        mask = (true_conf == 0.0)
        if mask.sum() > 0:
            loss_box = ops.complete_box_iou_loss(pred_boxes_xyxy[mask], true_boxes_xyxy[mask], reduction='mean')
            total_loss = (LAMBDA_CONF * loss_conf) + (LAMBDA_BOX * loss_box)
        else:
            loss_box = 0.0
            total_loss = LAMBDA_CONF * loss_conf
              
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()

    train_loss_avg = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0

    # ================= VALIDATION =================
    model.eval()

    val_loss = 0

    # Face / No-face accuracy
    val_correct = 0
    val_total = 0

    # IoU metric
    iou_sum = 0
    iou_count = 0

    # Detection accuracy
    det_correct = 0
    det_total = 0

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            pred_conf = outputs[:, 0]
            pred_boxes = torch.sigmoid(outputs[:,1:5])
            true_conf = labels[:, 0]
            true_boxes = labels[:, 1:5]

            pred_cx, pred_cy, pred_w, pred_h = pred_boxes[:,0], pred_boxes[:,1], pred_boxes[:,2], pred_boxes[:,3]
            true_cx, true_cy, true_w, true_h = true_boxes[:,0], true_boxes[:,1], true_boxes[:,2], true_boxes[:,3]

            pred_x1 = pred_cx - pred_w / 2
            pred_y1 = pred_cy - pred_h / 2
            pred_x2 = pred_cx + pred_w / 2
            pred_y2 = pred_cy + pred_h / 2
            pred_boxes_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1).clamp(0.0, 1.0)

            true_x1 = true_cx - true_w / 2
            true_y1 = true_cy - true_h / 2
            true_x2 = true_cx + true_w / 2
            true_y2 = true_cy + true_h / 2
            true_boxes_xyxy = torch.stack([true_x1, true_y1, true_x2, true_y2], dim=1).clamp(0.0, 1.0)

            # ===== Loss =====
            loss_conf = criterion(pred_conf, true_conf)

            mask = (true_conf == 0.0)

            if mask.sum() > 0:
                loss_box = ops.complete_box_iou_loss(pred_boxes_xyxy[mask], true_boxes_xyxy[mask], reduction='mean')
                v_loss = LAMBDA_CONF * loss_conf + LAMBDA_BOX * loss_box
            else:
                v_loss = LAMBDA_CONF * loss_conf

            val_loss += v_loss.item()

            # ===== Face / No-face accuracy =====
            probs = torch.sigmoid(pred_conf)
            preds = torch.where(probs > 0.5, 1.0, 0.0)

            val_correct += (preds == true_conf).sum().item()
            val_total += true_conf.size(0)

            # ===== IoU =====
            if mask.sum() > 0:
                iou_matrix = ops.box_iou(
                    pred_boxes_xyxy[mask],
                    true_boxes_xyxy[mask]
                )

                ious = iou_matrix.diag()

                iou_sum += ious.sum().item()
                iou_count += len(ious)

                # ===== Detection accuracy =====
                det_correct += (ious > 0.5).sum().item()
                det_total += len(ious)


    # ===== Statistics =====
    val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0

    val_loss_avg = (
        val_loss / len(val_loader)
        if len(val_loader) > 0 else 0.0
    )

    mean_iou = (
        iou_sum / iou_count
        if iou_count > 0 else 0.0
    )

    det_acc = (
        100 * det_correct / det_total
        if det_total > 0 else 0.0
    )

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss_avg:.4f}")
    print(
        f"Val Loss: {val_loss_avg:.4f} | "
        f"Conf Acc: {val_acc:.2f}% | "
        f"Mean IoU: {mean_iou:.4f} | "
        f"Detection Acc: {det_acc:.2f}%"
    )
    
    # ================= SAVE CHECKPOINT =================
    checkpoint_dir = os.path.join(current_dir, '..', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if mean_iou > best_iou:
        best_iou = mean_iou
        checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_iou': best_iou,
            'train_loss': train_loss_avg,
            'val_loss': val_loss_avg,
            'mean_iou': mean_iou
        }, checkpoint_path)
        print(f"✓ Checkpoint saved (Mean IoU: {mean_iou:.4f})")