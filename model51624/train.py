from ultralytics import YOLO

if __name__ ==  "__main__":
    model = YOLO('yolov8s.pt')
    results = model.train(data=r'model51624/data.yaml', epochs=200, imgsz=640, project=r"D:\Documents\Cos\Fourth Year\Second Semester\Final Thesis\models\model51624\results", device=0, val=True, batch=2, amp=False, optimize= True, lr0=0.001, lrf=0.01,)