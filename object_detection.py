"""
Object Detection Script using YOLOv8
ตรวจจับวัตถุทุกประเภทบนโลก (80+ classes from COCO dataset)
"""

import cv2
from ultralytics import YOLO
import argparse

def detect_objects(source='0', model_name='yolov8n.pt', conf_threshold=0.5):
    """
    ตรวจจับวัตถุจากกล้อง วิดีโอ หรือรูปภาพ

    Args:
        source: '0' สำหรับ webcam, หรือ path ของไฟล์ video/image
        model_name: ชื่อโมเดล YOLO (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
        conf_threshold: ค่าความมั่นใจขั้นต่ำ (0-1)
    """

    # โหลดโมเดล YOLO
    print(f"กำลังโหลดโมเดล {model_name}...")
    model = YOLO(model_name)

    # เปิด source (webcam/video/image)
    if source == '0':
        source = 0
        cap = cv2.VideoCapture(source)
        print("เปิดกล้องเรียบร้อย - กด 'q' เพื่อออก")
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            # ถ้าไม่ใช่วิดีโอ อาจเป็นรูปภาพ
            img = cv2.imread(source)
            if img is None:
                print(f"ไม่สามารถเปิดไฟล์: {source}")
                return

            # ตรวจจับวัตถุในรูปภาพ
            results = model(img, conf=conf_threshold)

            # แสดงผล
            annotated_frame = results[0].plot()
            cv2.imshow('Object Detection', annotated_frame)

            # แสดงรายการวัตถุที่ตรวจพบ
            print_detected_objects(results[0])

            print("\nกด 'q' เพื่อปิดหน้าต่าง")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

    # Loop สำหรับ video/webcam
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("ไม่สามารถอ่านเฟรมได้ หรือวิดีโอจบแล้ว")
            break

        # ตรวจจับวัตถุ
        results = model(frame, conf=conf_threshold, verbose=False)

        # วาดกรอบและข้อความบนภาพ
        annotated_frame = results[0].plot()

        # แสดงจำนวนวัตถุที่ตรวจพบ
        num_objects = len(results[0].boxes)
        cv2.putText(annotated_frame, f'Objects: {num_objects}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # แสดงผล
        cv2.imshow('YOLOv8 Object Detection', annotated_frame)

        # กด 'q' เพื่อออก
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ปิด
    cap.release()
    cv2.destroyAllWindows()
    print("ปิดโปรแกรมเรียบร้อย")


def print_detected_objects(result):
    """แสดงรายการวัตถุที่ตรวจพบ"""
    print("\n=== วัตถุที่ตรวจพบ ===")
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]
        print(f"- {class_name}: {confidence:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ตรวจจับวัตถุด้วย YOLOv8')
    parser.add_argument('--source', type=str, default='0',
                        help='แหล่งที่มา: "0" สำหรับ webcam, หรือ path ของไฟล์')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                        help='โมเดล YOLO (n=เร็วที่สุด, x=แม่นที่สุด)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='ค่าความมั่นใจขั้นต่ำ (0-1)')

    args = parser.parse_args()

    print("="*50)
    print("🎯 Object Detection - ตรวจจับวัตถุทุกอย่างบนโลก")
    print("="*50)
    print(f"แหล่งที่มา: {args.source}")
    print(f"โมเดล: {args.model}")
    print(f"Confidence threshold: {args.conf}")
    print("="*50)

    detect_objects(source=args.source, model_name=args.model, conf_threshold=args.conf)
