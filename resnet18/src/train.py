import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

class ResNet18(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.element_per_anchor = 4 + num_classes 
        out_channels = self.num_anchors * self.element_per_anchor  # Tổng số kênh (channels) đầu ra tại mỗi ô lưới
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )

        self.relu = nn.ReLU()
        
    # CỤM 1 (conv2_x): Biến đổi từ 64 kênh (112x112) -> 128 kênh (56x56)
        self.conv2_x_block1 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64)
        )
        self.conv2_x_block2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64)
        )
        
    # CỤM 2 (conv3_x): Biến đổi từ 64 kênh (56x56) -> 128 kênh (28x28)
        self.conv3_x_block1 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
        )
        self.conv3_x_block2 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 1, stride = 2),
            nn.BatchNorm2d(128)
        )

        
    # CỤM 3 (conv4_x): Biến đổi từ 128 kênh (28x28) -> 256 kênh (14x14)  
        self.conv4_x_block1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256)
        )
        self.conv4_x_block2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256)
        )
        self.downsample4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2),
            nn.BatchNorm2d(256)
        )

        
    # CỤM 4 (conv5_x): Biến đổi từ 256 kênh (14x14) -> 512 kênh (7x7)
        self.conv5_x_block1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )
        self.conv5_x_block2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )
        self.downsample5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2),
            nn.BatchNorm2d(512)
        )

        # ĐẦU DỰ ĐOÁN CHUẨN: Dùng Conv 1x1 thay thế hoàn toàn cho FC mạng tuyến tính
        self.prediction = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels = out_channels, kernel_size = 1) # Conv 1x1 quyết định số đầu ra
        )
        
        
    def forward(self, x):
        x = self.conv1(x)

    # CỤM 1
        # KHỐI 1
        identity = x                     # Nhánh phụ: Giữ nguyên bản sao của x làm đường tắt
        fx = self.conv2_x_block1(x)      # Nhánh chính: Tính toán qua khối 1
        x = self.relu(fx + identity)     # Cộng khối dư và kích hoạt ReLU -> Đây là đầu ra khối 1
        
        # KHỐI 2 
        identity = x                     # Nhánh phụ: Giữ nguyên đầu vào của khối 2
        fx = self.conv2_x_block2(x)      # Nhánh chính: Tính toán qua khối 2 với trọng số riêng
        x = self.relu(fx + identity)     # Cộng khối dư và kích hoạt ReLU -> Đầu ra tầng conv2_x
        
    # CỤM 2
        # KHỐI 1
        identity = self.downsample3(x)
        fx = self.conv3_x_block1(x)
        x = self.relu(fx + identity)

        # KHỐI 2
        identity = x
        fx = self.conv3_x_block2(x)
        x = self.relu(fx + identity)

        
    # CỤM 3 (conv4_x)
        # KHỐI 1
        identity = self.downsample4(x)
        fx = self.conv4_x_block1(x)
        x = self.relu(fx + identity)

        # KHỐI 2
        identity = x
        fx = self.conv4_x_block2(x)
        x = self.relu(fx + identity)
        
    # CỤM 4 (conv5_x)
        # KHỐI 1
        identity = self.downsample5(x)
        fx = self.conv5_x_block1(x)
        x = self.relu(fx + identity)

        # KHỐI 2
        identity = x
        fx = self.conv5_x_block2(x)
        features = self.relu(fx + identity)


        predictions = self.prediction(features)   # Kích thước hiện tại: [Batch_size, num_anchors * (4 + num_classes), H_grid, W_grid]

        batch_size, _, h, w = predictions.size()
        predictions = predictions.permute(0, 2, 3, 1).contiguous() 
        # Kích thước sau permute: [Batch_size, H_grid, W_grid, num_anchors * element_per_anchor]
        
        predictions = torch.flatten(predictions, start_dim=1)
        # Kích thước đầu ra cuối cùng: [Batch_size, 7 * 7 * num_anchors * (4 + num_classes)]

        return predictions
    

ResNet = ResNet18(num_classes = 20, num_anchors = 2)
test = torch.randn(1, 3, 224, 224)
ResNet(test)