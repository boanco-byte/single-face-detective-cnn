import os
import cv2
from typing import Dict
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )

        self.relu = nn.ReLU()
        
    # CỤM 1 (conv2_x): Biến đổi từ 64 kênh (112x112) -> 64 kênh (56x56)   
        self.conv2_x_block1 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(64)
        )
        self.conv2_x_block2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(64)
        )
    # [Batch_size, Channels = 64, Height = 56, Width = 56]
        
        
    # CỤM 2 (conv3_x): Biến đổi từ 64 kênh (56x56) -> 128 kênh (28x28)
        self.conv3_x_block1 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(128),
        )
        self.conv3_x_block2 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(128)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 1, stride = 2, bias = False),
            nn.BatchNorm2d(128)
        )
    # [Batch_size, Channels = 128, Height = 28, Width = 28]

        
    # CỤM 3 (conv4_x): Biến đổi từ 128 kênh (28x28) -> 256 kênh (14x14)  
        self.conv4_x_block1 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(256)
        )
        self.conv4_x_block2 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(256)
        )
        self.downsample4 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 1, stride = 2, bias = False),
            nn.BatchNorm2d(256)
        )
    # [Batch_size, Channels = 256, Height = 14, Width = 14]

        
    # CỤM 4 (conv5_x): Biến đổi từ 256 kênh (14x14) -> 512 kênh (7x7)
        self.conv5_x_block1 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(512)
        )
        self.conv5_x_block2 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(512)
        )
        self.downsample5 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 1, stride = 2, bias = False),
            nn.BatchNorm2d(512)
        )
    # [Batch_size, Channels = 512, Height = 7, Width = 7]
        

    def forward(self, x):
        x1 = self.conv1(x)

    # CỤM 1
        # KHỐI 1
        identity = x1                     # Nhánh phụ: Giữ nguyên bản sao của x làm đường tắt
        fx = self.conv2_x_block1(x1)      # Nhánh chính: Tính toán qua khối 1
        x2 = self.relu(fx + identity)     # Cộng khối dư và kích hoạt ReLU -> Đây là đầu ra khối 1
        
        # KHỐI 2 
        identity = x2                    # Nhánh phụ: Giữ nguyên đầu vào của khối 2
        fx = self.conv2_x_block2(x2)      # Nhánh chính: Tính toán qua khối 2 với trọng số riêng
        c2 = self.relu(fx + identity)     # Cộng khối dư và kích hoạt ReLU -> Đầu ra tầng conv2_x
    # c2.shape = [Batch_size, Channels = 64, Height = 56, Width = 56]

    # CỤM 2
        # KHỐI 1
        identity = self.downsample3(c2)
        fx = self.conv3_x_block1(c2)
        x3 = self.relu(fx + identity)

        # KHỐI 2
        identity = x3
        fx = self.conv3_x_block2(x3)
        c3 = self.relu(fx + identity)
    # c3.shape = [Batch_size, Channels = 128, Height = 28, Width = 28]
        
    # CỤM 3 (conv4_x)
        # KHỐI 1
        identity = self.downsample4(c3)
        fx = self.conv4_x_block1(c3)
        x4 = self.relu(fx + identity)

        # KHỐI 2
        identity = x4
        fx = self.conv4_x_block2(x4)
        c4 = self.relu(fx + identity)
    # c4.shape = [Batch_size, Channels = 256, Height = 14, Width = 14]
        
    # CỤM 4 (conv5_x)
        # KHỐI 1
        identity = self.downsample5(c4)
        fx = self.conv5_x_block1(c4)
        x5 = self.relu(fx + identity)

        # KHỐI 2
        identity = x5
        fx = self.conv5_x_block2(x5)
        c5 = self.relu(fx + identity)
    # c5.shape = [Batch_size, Channels = 512, Height = 7, Width = 7]
        
        return {
            "c2": c2,
            "c3": c3,
            "c4": c4,
            "c5": c5
        }


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.lat2 = nn.Conv2d(in_channels[0], out_channels, kernel_size = 1)
        self.lat3 = nn.Conv2d(in_channels[1], out_channels, kernel_size = 1)
        self.lat4 = nn.Conv2d(in_channels[2], out_channels, kernel_size = 1)
        self.lat5 = nn.Conv2d(in_channels[3], out_channels, kernel_size = 1)

        self.fpn_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.fpn_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.fpn_conv4 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.fpn_conv5 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        
        self.upsample = nn.Upsample(scale_factor = 2,  mode= 'nearest')

    def forward(self, inputs):
        c2, c3, c4, c5 = inputs["c2"], inputs["c3"], inputs["c4"], inputs["c5"]
        
        lat5 = self.lat5(c5)
        lat4 = self.lat4(c4)
        lat3 = self.lat3(c3)
        lat2 = self.lat2(c2)

        p5 = lat5
        p4 = self.upsample(p5) + lat4
        p3 = self.upsample(p4) + lat3
        p2 = self.upsample(p3) + lat2

        p5 = self.fpn_conv5(p5)
        p4 = self.fpn_conv4(p4)
        p3 = self.fpn_conv3(p3)
        p2 = self.fpn_conv2(p2)

        return {
            "p2": p2,
            "p3": p3,
            "p4": p4,
            "p5": p5
        }
           

class AnchorGenerator(nn.Module):
    def __init__(self, scales, ratios):
        super().__init__()
        self.scales = torch.tensor(scales)
        self.ratios = torch.tensor(ratios)

        self.base_anchors = nn.ParameterList()

        for scale in scales:
            width = scale * torch.sqrt(self.ratios)        # [num_ratios]
            height = scale / torch.sqrt(self.ratios)       # [num_ratios]

            anchor_shape = torch.stack([width, height], dim=-1) / 2              # [num_ratios, 2]
            base_anchor = torch.cat([-anchor_shape, anchor_shape], dim=-1)       # [num_ratios, 4]

            self.base_anchors.append(nn.Parameter(base_anchor.unsqueeze(dim=0), requires_grad=False))    
            # Thêm chiều để broadcast: [1, num_ratios, 4]

    def forward(self, feature_maps, strides):
        output_anchors: Dict[str, torch.Tensor] = {}

        # Dùng enumerate để lấy chỉ số level_idx làm khóa truy cập mảng strides
        # level_idx: Chỉ số của tầng FPN (0, 1, 2...)
        # level_name: Tên của tầng FPN (ví dụ: "p3", "p4", "p5"...)
        
        for level_idx, (level_name, feature_map) in enumerate(feature_maps.items()):
            device = feature_map.device
            grid_h, grid_w = feature_map.shape[-2:] 
            
            stride = strides[level_idx] 
            
            # Lấy anchor mẫu tương ứng với mức scale của level_idx này: [1, num_ratios, 4]
            base_anchor = self.base_anchors[level_idx].to(device)              # [1, num_ratios, 4]


            cx = (torch.arange(grid_w, device=device) + 0.5) * stride          # [grid_w]
            cy = (torch.arange(grid_h, device=device) + 0.5) * stride          # [grid_h]

            grid_y, grid_x = torch.meshgrid(cy, cx, indexing='ij')             #[grid_h, grid_w]
            centers = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)     # [grid_h * grid_w, 2]
            centers = centers.repeat(1, 2).unsqueeze(dim=1)                    # [grid_h * grid_w, 1, 4]
            
            # Broadcast: [grid_h * grid_w, 1, 4] + [1, num_ratios, 4]
            level_anchors = centers + base_anchor                              # [grid_h * grid_w, num_ratios, 4]
            level_anchors = level_anchors.reshape(-1, 4)                       # [grid_h * grid_w * num_ratios, 4]

            output_anchors[level_name] = level_anchors

        return output_anchors


class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1),     #[Batch_size, in_channels, Height, Width]
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size = 1, padding = 0)  
        # [Batch_size, num_anchors * num_classes, Height, Width]    Thông thường số channels sẽ là 2 * num_anchors

        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size = 1, padding = 0)
        # [Batch_size, num_anchors * 4, Height, Width]    (dx, dy, dw, dh)

    def forward(self, feature_maps):
        output_cls_logits: Dict[str, torch.Tensor] = {}
        output_bbox_pred: Dict[str, torch.Tensor] = {}
            
        for level_idx, (level_name, feature_map) in enumerate(feature_maps.items()):
            B, C, H, W = feature_map.shape
            
            conv = self.conv(feature_map)

            cls_logits = self.cls_logits(conv)
            bbox_pred = self.bbox_pred(conv)

            # Đổi trục từ [Batch_size, Channels, Height, Width] -> [Batch_size, Height, Width, Channels]
            cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()

            # cls_logits.Shape từ [Batch_size, Height, Width, num_anchors * num_classes] -> [Batch_size, Height * Width * num_anchors, num_classes]
            # bbox_pred.Shape từ [Batch_size, Height, Width, num_anchors * 4] -> [Batch_size, Height * Width * num_anchors, 4]
            cls_logits = cls_logits.reshape(B, -1, self.num_classes)
            bbox_pred = bbox_pred.reshape(B, -1, 4)
                
            output_cls_logits[level_name] = cls_logits
            output_bbox_pred[level_name] = bbox_pred

        return output_cls_logits, output_bbox_pred

class RPN(nn.Module):
    def __init__(self):
        super().__init__()



    def decode(anchors_all, bbox_preds_all, width, height):
        x1, y1, x2, y2 = torch.chunk(anchors_all, chunks=4, dim=-1)
        dx, dy, dw, dh = torch.chunk(bbox_preds_all, chunks=4, dim=-1)  

        anchor_w = x2 - x1
        anchor_h = y2 - y1
        anchor_x = (x1 + x2) / 2
        anchor_y = (y1 + y2) / 2
        
        new_x = anchor_x + dx * anchor_w
        new_y = anchor_y + dy * anchor_h
        new_w = anchor_w * torch.exp(dw)
        new_h = anchor_h * torch.exp(dh)
        
        new_x1 = torch.clamp((new_x - new_w / 2), min = 0, max = width)
        new_y1 = torch.clamp((new_y - new_h / 2), min = 0, max = height)
        new_x2 = torch.clamp((new_x + new_w / 2), min = 0, max = width)
        new_y2 = torch.clamp((new_y + new_h / 2), min = 0, max = height)

        proposal_boxes = torch.cat([new_x1, new_y1, new_x2, new_y2], dim = -1)
        return proposal_boxes

    def remove_small_box(proposal_boxes, cls_logits_all, mini_size):
        x1, y1, x2, y2 = torch.chunk(proposal_boxes, chunks = 4, dim = -1)

        width = x2 - x1
        height = y2 - y1

        mask = (width > mini_size) & (height > mini_size)

        scores = cls_logits_all[..., 1]                          # Lọc cột object
        
        # Nếu thỏa mãn điều kiện (mask=True) thì giữ nguyên obj_logits, ngược lại (False) thì gán -inf
        scores = torch.where(mask.squeeze(dim = -1), scores, float('-inf'))

        return proposal_boxes, scores

    def select_top_k_proposals(proposal_boxes, scores, k):
        k = min(k, proposal_boxes.shape[1])
        
        top_scores, indices = torch.topk(scores, k = k, dim = -1)

        bbox_indices = indices.unsqueeze(dim = -1).expand(-1, -1, 4)
        proposal_boxes = torch.gather(proposal_boxes, dim = 1, index = bbox_indices)

        return top_scores, proposal_boxes



# Test

scales = [32, 64, 128, 256]
ratios = [0.5, 1.0, 2.0]
strides_list = [4, 8, 16, 32] # KHAI BÁO DẠNG MẢNG THEO Ý BẠN

width = height = 224

# Khởi tạo mô hình
generator = AnchorGenerator(scales=scales, ratios=ratios)
rpnhead = RPNHead(256, 3, 2)
# Tạo dữ liệu giả lập (Dictionary feature_maps đầu ra từ FPN)
fake_feature_maps = {
    "p2": torch.randn(5, 256, 56, 56),
    "p3": torch.randn(5, 256, 28, 28),
    "p4": torch.randn(5, 256, 14, 14),
    "p5": torch.randn(5, 256, 7, 7),
}

# Gọi hàm forward truyền vào strides dạng List
anchors = generator(fake_feature_maps, strides_list)
cls_logits, bbox_preds = rpnhead(fake_feature_maps)

anchors_all = torch.cat(list(anchors.values()), dim=-2)
cls_logits_all = torch.cat(list(cls_logits.values()), dim=-2)
bbox_preds_all = torch.cat(list(bbox_preds.values()), dim=-2)

print(anchors_all.shape)
print(cls_logits_all.shape)
print(bbox_preds_all.shape)

x1, y1, x2, y2 = torch.chunk(anchors_all, chunks=4, dim=-1)
dx, dy, dw, dh = torch.chunk(bbox_preds_all, chunks=4, dim=-1)

anchor_w = x2 - x1
anchor_h = y2 - y1

anchor_x = (x1 + x2) / 2
anchor_y = (y1 + y2) / 2

new_x = anchor_x + dx * anchor_w
new_y = anchor_y + dy * anchor_h

new_w = anchor_w * torch.exp(dw)
new_h = anchor_h * torch.exp(dh)

new_x1 = torch.clamp((new_x - new_w / 2), min = 0, max = width)
new_y1 = torch.clamp((new_y - new_h / 2), min = 0, max = height)
new_x2 = torch.clamp((new_x + new_w / 2), min = 0, max = width)
new_y2 = torch.clamp((new_y + new_h / 2), min = 0, max = height)

proposal_boxes = torch.cat([new_x1, new_y1, new_x2, new_y2], dim = 2)
print("Kich thuoc cua proposal_boxes: ", proposal_boxes.shape)
size = proposal_boxes.shape

widths = new_x2 - new_x1
heights = new_y2 - new_y1

print(widths.shape)
print(heights.shape)
mini_size = 16
mask = ((widths.squeeze() >= mini_size) & (heights.squeeze() >= mini_size))
print("Kich thuoc cua mask: ", mask.shape)
"""
proposals = proposal_boxes[mask]
scores = cls_logits_all[...,0][mask]
bbox_preds = bbox_preds_all[mask]

print("Kich thuoc cua scores: ", scores.shape)
print("Kich thuoc cua bbox_pred: ", bbox_preds.shape)
print("Kich thuoc cua proposal: ", proposals.shape)
"""
TOP_K = 2000
TOP_K = min(TOP_K, size[1])

cls_logits_top, indices = torch.topk(cls_logits_all[..., 0:1], k = TOP_K, dim = -2)   #cls_logits_all co kich thuoc [Batch, Nums, classes = 2)
print("Kich thuoc cua indices: ", indices.shape)
bbox_indices = indices.expand(-1, -1, 4)
bbox_preds_top = torch.gather(bbox_preds_all, dim=-2, index=bbox_indices)
proposal_boxes_top = torch.gather(proposal_boxes, dim=-2, index=bbox_indices)

print("Kich thuoc cua top cls_logits: ", cls_logits_top.shape)
print("Kich thuoc cua top bbox_pred: ", bbox_preds_top.shape)
print("Kich thuoc cua top proposal_boxes: ", proposal_boxes_top.shape)


print("Kich thuoc new_x: ", new_x1.shape, " va so chieu: ", new_x.ndim)
print("Kich thuoc new_y: ", new_y1.shape, " va so chieu: ", new_y.ndim)
print("Kich thuoc new_w: ", new_x2.shape, " va so chieu: ", new_w.ndim)
print("Kich thuoc new_h: ", new_y2.shape, " va so chieu: ", new_h.ndim)
