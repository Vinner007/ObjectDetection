"""
YOLOv11 Object Detection - เวอร์ชันล่าสุด แม่นกว่า เร็วกว่า
"""

import cv2
from ultralytics import YOLO
import argparse

def detect_objects_v11(source='0', model_name='yolo11x.pt', conf_threshold=0.5):
    """
    ตรวจจับวัตถุด้วย YOLOv11 (เวอร์ชันล่าสุด)

    โมเดล YOLOv11:
    - yolo11n.pt: เร็วสุด (2.6M parameters)
    - yolo11s.pt: เล็ก (9.4M parameters)
    - yolo11m.pt: กลาง (20.1M parameters)
    - yolo11l.pt: ใหญ่ (25.3M parameters)
    - yolo11x.pt: แม่นสุด โหดสุด (56.9M parameters) ⭐
    """

    print(f"กำลังโหลด YOLOv11 โมเดล {model_name}...")
    model = YOLO(model_name)

    # เปิด source
    if source == '0':
        source = 0
        cap = cv2.VideoCapture(source)
        print("เปิดกล้องเรียบร้อย - กด 'q' เพื่อออก")
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            img = cv2.imread(source)
            if img is None:
                print(f"ไม่สามารถเปิดไฟล์: {source}")
                return

            results = model(img, conf=conf_threshold)
            annotated_frame = results[0].plot()

            # แสดงข้อมูลวัตถุที่พบ
            print_detailed_results(results[0])

            cv2.imshow('YOLOv11 Detection', annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

    # Real-time detection
    import time
    prev_time = 0
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # คำนวณ FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        frame_count += 1

        # ตรวจจับวัตถุ
        results = model(frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()

        # แสดงข้อมูล
        num_objects = len(results[0].boxes)
        cv2.putText(annotated_frame, f'YOLOv11x - FPS: {fps:.1f}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Objects: {num_objects}',
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # แสดงรายละเอียดทุก 30 เฟรม
        if frame_count % 30 == 0 and num_objects > 0:
            print(f"\n--- Frame {frame_count} ---")
            print_detailed_results(results[0])

        cv2.imshow('YOLOv11 Object Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("ปิดโปรแกรมเรียบร้อย")


def print_detailed_results(result):
    """แสดงรายละเอียดวัตถุที่ตรวจพบ"""
    print("="*50)
    for i, box in enumerate(result.boxes, 1):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]

        # พิกัดกรอบ (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        print(f"{i}. {class_name}")
        print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"   Position: ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv11 Object Detection - เวอร์ชันโหดสุด')
    parser.add_argument('--source', type=str, default='0',
                        help='แหล่งที่มา: "0" สำหรับ webcam')
    parser.add_argument('--model', type=str, default='yolo11x.pt',
                        choices=['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt',
                                'yolo11l.pt', 'yolo11x.pt'],
                        help='โมเดล YOLOv11')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='ค่าความมั่นใจขั้นต่ำ')

    args = parser.parse_args()

    print("="*60)
    print("🔥 YOLOv11 - State-of-the-Art Object Detection")
    print("="*60)
    print(f"โมเดล: {args.model}")
    print(f"Confidence threshold: {args.conf}")
    print("="*60)

    detect_objects_v11(source=args.source, model_name=args.model,
                       conf_threshold=args.conf)
