

from ultralytics import YOLO

#First we uesd linemodel = YOLO("yolov8n.yaml")
#Then we load the trained model to train it bettet again.
# Load a model
model = YOLO('/Users/dd/ikt213g23h/assignments/solutions/cattle-count/runs/detect/train38/weights/best.pt')#load trainedmodel



#set customized parameters if you want. or use defaults from yolo
# Use the model
results = model.train(
        data="datasett.yaml",
        epochs=300
        #  batch=16,
        # lrf=0.001,
        #momentum=0.9,
        #weight_decay=0.0005

)  # train the model