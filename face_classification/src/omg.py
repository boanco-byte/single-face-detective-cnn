import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '..', 'dataset/train/labels/000_1OC3DT_jpg.rf.8SZFvcGh0g8DpJQk1NJy.txt')

with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Tách toàn bộ nội dung file thành một danh sách các phần tử
elements = content.split() # Dùng split() không tham số sẽ tự động tách theo mọi khoảng trắng/xuống dòng

# Duyệt qua từng phần tử để in kiểu dữ liệu
for item in elements:
    print(f"Phần tử: '{item}' | Kiểu dữ liệu: {type(item)}")
