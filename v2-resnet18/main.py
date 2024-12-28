import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO
import gradio as gr

# Kiểm tra thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Sử dụng thiết bị: {device}')

# Định nghĩa kiến trúc mô hình phân loại
class ClassificationModel(nn.Module):
    def __init__(self, num_classes=12):
        super(ClassificationModel, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Sử dụng pretrained weights
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)  # Thay đổi lớp cuối cùng

    def forward(self, x):
        return self.model(x)

# Khởi tạo và tải mô hình phân loại
classification_model = ClassificationModel(num_classes=12)
classification_model.load_state_dict(torch.load('classification_Resnet18.pt', map_location=device))
classification_model.to(device)
classification_model.eval()

# Tải mô hình YOLO
yolo_model = YOLO('best.pt')  # Đảm bảo rằng 'best.pt' nằm trong thư mục hiện tại

# Định nghĩa các lớp mụn
class_labels = [
    'acne_scars', 'blackhead', 'cystic', 'flat_wart', 'folliculitis',
    'keloid', 'milium', 'papular', 'purulent', 'sebo-crystan-conglo',
    'syringoma', 'whitehead'
]

# Ánh xạ tiếng Anh -> tiếng Việt
class_mapping = {
    'acne_scars': 'Sẹo mụn',
    'blackhead': 'Mụn đầu đen',
    'cystic': 'Mụn nang',
    'flat_wart': 'Mụn sần phẳng',
    'folliculitis': 'Viêm nang lông',
    'keloid': 'Mụn sẹo uốn',
    'milium': 'Mụn mili',
    'papular': 'Mụn nhỏ',
    'purulent': 'Mụn mủ',
    'sebo-crystan-conglo': 'Mụn bã đen kết tủa',
    'syringoma': 'Mụn nang mồ hôi',
    'whitehead': 'Mụn đầu trắng'
}

# Định nghĩa các biến đổi dữ liệu cho mô hình phân loại
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def detect_and_classify(image, image_size=640, conf_threshold=0.4, iou_threshold=0.5):
    """
    Hàm này nhận vào một ảnh, phát hiện các vùng mụn bằng YOLO,
    phân loại từng vùng mụn bằng mô hình ResNet18, và trả về ảnh đã được
    annotate cùng với các thông tin liên quan.
    """
    # Mở ảnh và chuyển đổi sang RGB
    pil_image = Image.open(image).convert("RGB")

    # Dự đoán bằng YOLO
    results = yolo_model.predict(pil_image, conf=conf_threshold, iou=iou_threshold, imgsz=image_size)
    boxes = results[0].boxes
    num_boxes = len(boxes)

    if num_boxes == 0:
        severity = "Tốt"
        recommendation = "Làn da bạn khá ổn! Tiếp tục duy trì thói quen chăm sóc da."
        return pil_image, f"Tình trạng mụn: {severity}", recommendation, "Không phát hiện mụn."

    # Lấy thông tin bounding boxes
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)  # Toạ độ bounding box
    confidences = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)

    # Chuẩn bị vẽ
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Danh sách để lưu kết quả phân loại
    classified_results = []

    for i, (box, cls_id, conf) in enumerate(zip(xyxy, class_ids, confidences), start=1):
        x1, y1, x2, y2 = box
        class_name_en = class_labels[cls_id]
        class_name_vn = class_mapping.get(class_name_en, class_name_en)

        # Cắt crop vùng mụn
        crop = pil_image.crop((x1, y1, x2, y2))
        img_transformed = transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            output = classification_model(img_transformed)
            probabilities = torch.softmax(output, dim=1)
            top_prob, top_class = probabilities.topk(1, dim=1)
            top_prob = top_prob.item()
            top_class = top_class.item()
            class_name_en = class_labels[top_class]
            class_name_vn = class_mapping.get(class_name_en, class_name_en)

        # Vẽ bounding box và nhãn
        label = f"#{i}: {class_name_en} ({class_name_vn}) ({top_prob:.2f})"
        # Sử dụng textbbox thay vì textsize
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill="red")
        draw.text((x1, y1 - text_h), label, fill="white", font=font)

        # Thêm kết quả phân loại vào danh sách
        classified_results.append((i, class_name_en, class_name_vn))

    # Đánh giá tình trạng da dựa trên số lượng mụn
    if num_boxes > 20:
        severity = "Nặng"
        recommendation = "Bạn nên đến gặp bác sĩ da liễu và sử dụng liệu trình trị mụn chuyên sâu."
    elif 10 <= num_boxes <= 20:
        severity = "Trung bình"
        recommendation = "Duy trì skincare đều đặn với sữa rửa mặt dịu nhẹ và dưỡng ẩm."
    else:
        severity = "Tốt"
        recommendation = "Làn da bạn khá ổn! Tiếp tục duy trì thói quen chăm sóc da."

    # Liệt kê loại mụn
    acne_types_str = "Danh sách mụn phát hiện:\n"
    for idx, cname_en, cname_vn in classified_results:
        acne_types_str += f"Mụn #{idx}: {cname_en} ({cname_vn})\n"

    return pil_image, f"Tình trạng mụn: {severity}", recommendation, acne_types_str

# Mô tả ứng dụng
description_md = """
## Ứng Dụng Nhận Diện và Phân Loại Mụn Bằng YOLO và ResNet18

1. **Phát hiện mụn:** Sử dụng mô hình YOLO để phát hiện các vùng mụn trên khuôn mặt.
2. **Phân loại mụn:** Sử dụng mô hình ResNet18 đã được huấn luyện để phân loại từng vùng mụn thành 12 loại khác nhau bao gồm: acne_scars, blackhead, cystic, flat_wart, folliculitis, keloid, milium, papular, purulent, sebo-crystan-conglo, syringoma, whitehead
3. **Hiển thị kết quả:** Ảnh sau khi xử lý sẽ hiển thị các bounding boxes, nhãn tiếng Anh và tiếng Việt của loại mụn, cùng với độ chính xác của mỗi phân loại.
4. **Đánh giá tình trạng da:** Cung cấp đánh giá tổng quát về tình trạng da và khuyến nghị tương ứng dựa trên số lượng mụn được phát hiện.
"""

# Định nghĩa giao diện Gradio
inputs = [
    gr.Image(type="filepath", label="Ảnh Khuôn Mặt"),
    gr.Slider(minimum=320, maximum=1280, step=32, value=640, label="Kích thước ảnh (Image Size)"),
    gr.Slider(minimum=0, maximum=1, step=0.05, value=0.4, label="Ngưỡng Confidence"),
    gr.Slider(minimum=0, maximum=1, step=0.05, value=0.5, label="Ngưỡng IOU")
]

outputs = [
    gr.Image(type="pil", label="Ảnh Sau Khi Xử Lý"),
    gr.Textbox(label="Tình Trạng Mụn"),
    gr.Textbox(label="Khuyến Nghị"),
    gr.Textbox(label="Loại Mụn Phát Hiện")
]

# Tạo giao diện Gradio
app = gr.Interface(
    fn=detect_and_classify,
    inputs=inputs,
    outputs=outputs,
    title="YOLO + ResNet18 Phát Hiện và Phân Loại Mụn",
    description=description_md
)

# Khởi chạy ứng dụng
app.launch(share=True)
