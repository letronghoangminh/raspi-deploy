import torch
import cv2

class YoloV5:
    def __init__(self, weight_path, image_path, model_path="ultralytics/yolov5"):
        self.model_path = model_path
        self.weight_path = weight_path
        self.image_path = image_path

    def load_model(self):
        model = torch.hub.load(self.model_path, "custom", self.weight_path)
        return model

    def predict_labels(self):
        model = self.load_model()
        final_results = []
        results = model(self.image_path)
        for i, row in results.pandas().xyxy[0].iterrows():
            confidence = float(row["confidence"])
            leaves_type = str(row["name"])
            xmin, ymin, xmax, ymax = (
                int(row["xmin"]),
                int(row["ymin"]),
                int(row["xmax"]),
                int(row["ymax"]),
            )
            final_results.append(
                (leaves_type, confidence, xmin, ymin, xmax, ymax))
        return final_results
    
    def inference(self):
        results = self.predict_labels()
        img = cv2.imread('static/test.jpeg')
        leaves_type = results[0][0]
        confidence = results[0][1]
        xmin = results[0][2]
        ymin = results[0][3]
        xmax = results[0][4]
        ymax = results[0][5]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        cv2.putText(
                img,
                str(round(confidence, 2)),
                (xmin, ymin + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )
        cv2.imwrite("static/inference.jpeg", img)
