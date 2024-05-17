from roboflow import Roboflow
rf = Roboflow(api_key="oMzKbwQ1H9vuWVtr8HPF")
project = rf.workspace("for-thesis-46cfa").project("vehicle-detection-ph-nt1ns")
version = project.version(5)
dataset = version.download("yolov8")
