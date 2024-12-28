import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO
import gradio as gr

# ---------------------------
# 1. Kiểm tra thiết bị
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

# ---------------------------
# 2. Định nghĩa kiến trúc mô hình phân loại (DenseNet121)
# ---------------------------
class ClassificationModel(nn.Module):
    def __init__(self, num_classes=12, dropout_p=0.5):
        super(ClassificationModel, self).__init__()
        # Sử dụng DenseNet121 pretrained
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        # Lấy số features đầu ra của classifier
        in_feats = self.model.classifier.in_features
        # Thay thế classifier cuối
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_feats, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# ---------------------------
# 3. Khởi tạo và tải mô hình phân loại
# ---------------------------
classification_model = ClassificationModel(num_classes=12, dropout_p=0.5)
classification_model.load_state_dict(torch.load("classification_DenseNet121.pt", map_location=device))
classification_model.to(device)
classification_model.eval()

# ---------------------------
# 4. Tải mô hình YOLO
# ---------------------------
# Chỉnh tên file .pt tuỳ theo mô hình YOLO của bạn
yolo_model = YOLO("best.pt")  # hoặc 'yolov8n.pt' tùy bạn

# ---------------------------
# 5. Định nghĩa các lớp mụn
# ---------------------------
class_labels = [
    "acne_scars",
    "blackhead",
    "cystic",
    "flat_wart",
    "folliculitis",
    "keloid",
    "milium",
    "papular",
    "purulent",
    "sebo-crystan-conglo",
    "syringoma",
    "whitehead"
]

# ---------------------------
# 6. Ánh xạ tiếng Anh -> tiếng Việt (tuỳ chỉnh)
# ---------------------------
class_mapping = {
    "acne_scars": "Sẹo mụn",
    "blackhead": "Mụn đầu đen",
    "cystic": "Mụn nang",
    "flat_wart": "Mụn sần phẳng",
    "folliculitis": "Viêm nang lông",
    "keloid": "Sẹo lồi/keloid",
    "milium": "Mụn mili",
    "papular": "Mụn nhỏ",
    "purulent": "Mụn mủ",
    "sebo-crystan-conglo": "Mụn bã đen kết tủa",
    "syringoma": "Mụn u ống tuyến mồ hôi",
    "whitehead": "Mụn đầu trắng"
}

# ---------------------------
# 7. Định nghĩa các biến đổi dữ liệu cho mô hình DenseNet
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------
# 8. Hàm phát hiện và phân loại
# ---------------------------
def detect_and_classify(image, image_size=640, conf_threshold=0.4, iou_threshold=0.5):
    """
    - Phát hiện vùng mụn bằng YOLO.
    - Mỗi vùng bounding box được cắt và đưa qua DenseNet121 để phân loại.
    - Vẽ box và nhãn (tiếng Anh + tiếng Việt + confidence) lên ảnh, rồi trả về.
    - Đồng thời đánh giá tình trạng da và khuyến nghị.
    """
    # Đọc ảnh PIL, chuyển sang RGB (phòng trường hợp ảnh là RGBA)
    pil_image = Image.open(image).convert("RGB")

    # Dự đoán bằng YOLO
    results = yolo_model.predict(
        source=pil_image,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=image_size
    )
    boxes = results[0].boxes
    num_boxes = len(boxes)

    # Nếu không phát hiện mụn nào
    if num_boxes == 0:
        severity = "Tốt"
        recommendation = "Làn da bạn khá ổn! Tiếp tục duy trì thói quen chăm sóc da."
        return pil_image, f"Tình trạng mụn: {severity}", recommendation, "Không phát hiện mụn."

    # Lấy thông tin bounding boxes
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)  # toạ độ (x1,y1,x2,y2)
    confidences = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)

    # Chuẩn bị vẽ
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        # Nếu không có font arial, dùng font mặc định
        font = ImageFont.load_default()

    # Danh sách lưu kết quả phân loại
    classified_results = []

    for i, (box, cls_id, conf) in enumerate(zip(xyxy, class_ids, confidences), start=1):
        x1, y1, x2, y2 = box

        # Crop vùng mụn
        crop = pil_image.crop((x1, y1, x2, y2))
        img_transformed = transform(crop).unsqueeze(0).to(device)

        # Phân loại bằng DenseNet121
        with torch.no_grad():
            outputs = classification_model(img_transformed)
            probabilities = torch.softmax(outputs, dim=1)
            top_prob, top_class = probabilities.topk(1, dim=1)
            top_prob = top_prob.item()
            top_class = top_class.item()

            class_name_en = class_labels[top_class]
            class_name_vn = class_mapping.get(class_name_en, class_name_en)

        # Vẽ bounding box và nhãn
        label = f"#{i}: {class_name_en} ({class_name_vn}) ({top_prob:.2f})"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Vẽ hình chữ nhật
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        # Vẽ nền cho text
        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill="red")
        # Vẽ text
        draw.text((x1, y1 - text_h), label, fill="white", font=font)

        # Thêm kết quả phân loại vào danh sách
        classified_results.append((i, class_name_en, class_name_vn))

    # Đánh giá tình trạng da đơn giản dựa trên số vùng mụn
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

# ---------------------------
# 9. Mô tả ứng dụng
# ---------------------------
description_md = r"""
## Ứng Dụng Nhận Diện và Phân Loại Mụn Bằng YOLO và DenseNet121

1. **Phát hiện mụn**: Sử dụng mô hình YOLO để tìm các vùng mụn trên khuôn mặt.  
2. **Phân loại mụn**: Sau đó cắt từng vùng bounding box và đưa qua DenseNet121 để phân loại thành 12 loại.  
3. **Hiển thị kết quả**: Ảnh được vẽ bounding box và nhãn (tiếng Anh, tiếng Việt, xác suất).  
4. **Đánh giá tình trạng da**: Dựa trên tổng số vùng mụn phát hiện.  
5. **Khuyến nghị**: Thói quen skincare cơ bản hoặc đi khám da liễu nếu tình trạng nặng.
"""

# ---------------------------
# 10. Định nghĩa giao diện Gradio
# ---------------------------
# Input
inputs = [
    gr.Image(type="filepath", label="Ảnh Khuôn Mặt"),
    gr.Slider(minimum=320, maximum=1280, step=32, value=640, label="Kích thước ảnh (Image Size)"),
    gr.Slider(minimum=0, maximum=1, step=0.05, value=0.4, label="Ngưỡng Confidence"),
    gr.Slider(minimum=0, maximum=1, step=0.05, value=0.5, label="Ngưỡng IOU")
]

# Output
outputs = [
    gr.Image(type="pil", label="Ảnh Sau Khi Xử Lý"),
    gr.Textbox(label="Tình Trạng Mụn"),
    gr.Textbox(label="Khuyến Nghị"),
    gr.Textbox(label="Loại Mụn Phát Hiện")
]

# Tạo giao diện
app = gr.Interface(
    fn=detect_and_classify,
    inputs=inputs,
    outputs=outputs,
    title="YOLO + DenseNet121 Phát Hiện và Phân Loại Mụn",
    description=description_md
)

# ---------------------------
# 11. Khởi chạy ứng dụng
# ---------------------------
if __name__ == "__main__":
    app.launch(share=True)
