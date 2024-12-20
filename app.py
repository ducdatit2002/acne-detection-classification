import gradio as gr
import torch
from ultralyticsplus import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Tải model YOLO (detection)
yolo_model_path = "best.pt"
yolo_model = YOLO(yolo_model_path)

# Tải model classification (.h5)
classification_model_path = "classification.h5"
class_model = load_model(classification_model_path)

# Danh sách lớp theo thứ tự model classification
class_labels = ['Acne', 'Comedones', 'Papule', 'Pustule']

# Ánh xạ tiếng Anh -> tiếng Việt
class_mapping = {
    'Acne': 'Mụn trứng cá',
    'Comedones': 'Mụn cám',
    'Papule': 'Mụn sẩn',
    'Pustule': 'Mụn mủ'
}

def detect_and_classify(image, image_size, conf_threshold=0.4, iou_threshold=0.5):
    # image là đường dẫn file
    pil_image = Image.open(image).convert("RGB")

    # Detect vùng mụn với YOLO
    results = yolo_model.predict(pil_image, conf=conf_threshold, iou=iou_threshold, imgsz=image_size)
    boxes = results[0].boxes
    num_boxes = len(boxes)

    if num_boxes == 0:
        severity = "Tốt"
        recommendation = "Làn da bạn khá ổn! Tiếp tục duy trì thói quen chăm sóc da."
        return image, f"Tình trạng mụn: {severity}", recommendation, "Không phát hiện mụn."

    # Lấy toạ độ bounding box
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy()

    # Chuẩn bị vẽ
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()

    # Classification từng box
    classified_results = []
    for i, box in enumerate(xyxy, start=1):
        x1, y1, x2, y2 = box
        crop = pil_image.crop((x1, y1, x2, y2))

        # Tiền xử lý crop để đưa vào classification model
        crop_resized = crop.resize((224, 224))
        img_arr = img_to_array(crop_resized) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        # Dự đoán loại mụn
        pred = class_model.predict(img_arr)
        class_idx = np.argmax(pred)
        class_name_en = class_labels[class_idx]
        class_name_vn = class_mapping.get(class_name_en, class_name_en)  # Lấy tên tiếng Việt
        conf = confidences[i-1]

        # Vẽ box và label
        text = f"#{i}: {class_name_en} ({class_name_vn}) ({conf:.2f})"
        bbox = draw.textbbox((0,0), text, font=font)
        text_w = bbox[2]-bbox[0]
        text_h = bbox[3]-bbox[1]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill="red")
        draw.text((x1, y1 - text_h), text, fill="white", font=font)

        classified_results.append((i, class_name_en, class_name_vn))

    # Đánh giá tình trạng dựa trên số lượng mụn
    if num_boxes > 10:
        severity = "Nặng"
        recommendation = "Bạn nên đến gặp bác sĩ da liễu và sử dụng liệu trình trị mụn chuyên sâu."
    elif 5 <= num_boxes <= 10:
        severity = "Trung bình"
        recommendation = "Duy trì skincare đều đặn với sữa rửa mặt dịu nhẹ và dưỡng ẩm."
    else:
        severity = "Tốt"
        recommendation = "Làn da bạn khá ổn! Tiếp tục duy trì thói quen chăm sóc da."

    # Liệt kê loại mụn
    acne_types_str = "Danh sách mụn phát hiện:\n"
    for idx, cname_en, cname_vn in classified_results:
        acne_types_str += f"Mụn #{idx}: {cname_en} ({cname_vn})\n"

    # Lưu ảnh kết quả
    predicted_image_save_path = "predicted_image.jpg"
    pil_image.save(predicted_image_save_path)

    return predicted_image_save_path, f"Tình trạng mụn: {severity}", recommendation, acne_types_str

description_md = """
## Ứng dụng Nhận Diện (YOLO) & Phân Loại Mụn (Classification.h5)
1. Dùng YOLO để phát hiện vùng mụn trên khuôn mặt.
2. Dùng model classification (h5) để phân loại loại mụn.
3. Hiển thị kết quả lên ảnh cùng tình trạng da và gợi ý, kèm tên tiếng Anh và tiếng Việt của loại mụn.
"""

inputs = [
    gr.Image(type="filepath", label="Ảnh Khuôn Mặt"),
    gr.Slider(minimum=320, maximum=1280, step=32, value=640, label="Kích thước ảnh (Image Size)"),
    gr.Slider(minimum=0, maximum=1, step=0.05, value=0.4, label="Ngưỡng Confidence"),
    gr.Slider(minimum=0, maximum=1, step=0.05, value=0.5, label="Ngưỡng IOU")
]

outputs = [
    gr.Image(type="filepath", label="Ảnh Sau Khi Xử Lý"),
    gr.Textbox(label="Tình Trạng Mụn", interactive=False),
    gr.Textbox(label="Khuyến Nghị", interactive=False),
    gr.Textbox(label="Loại Mụn Phát Hiện", interactive=False)
]

app = gr.Interface(
    fn=detect_and_classify,
    inputs=inputs,
    outputs=outputs,
    title="YOLO + Classification (H5) Mụn Tiếng Việt",
    description=description_md
)

app.launch(share=True)
