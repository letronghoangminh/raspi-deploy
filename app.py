import cv2
import os
import requests

from flask import Flask, render_template,Response, redirect, url_for, request
from modules import YoloV5

app=Flask(__name__,  static_url_path='/static', static_folder='static')
camera = None
server = os.environ['SEARCH_SERVER']

def generate_frames():
    global camera
    camera = cv2.VideoCapture(0)
    
    while True:
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        #camera.release()
        #cv2.destroyAllWindows() 
        yield(b'--frame\r\n'
                   b'Con6tent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def get_frame():
    global camera
    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
            cv2.imwrite("static/test.jpeg", frame)
            break


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/take-pic')
def take_pic():
    get_frame()
    return redirect(url_for('predict'))

@app.route('/predict')
def predict():
    yolov5 = YoloV5(weight_path = "weights/best.onnx", image_path = "static/test.jpeg")
    final_results = yolov5.predict_labels()
    yolov5.inference()
    return render_template('inference.html', label=final_results[0][0], accuracy=(final_results[0][1] * 100), image='static/inference.jpeg')

@app.route('/search', methods=['POST'])
def search():
    label = request.form.get('label', 'label')
    res = requests.get(f'{server}/search/{label}')
    articles = res.json()

    return render_template('result.html', articles=articles)

@app.route('/corpus/<index>', methods=['GET'])
def detail(index):
    res = requests.get(f'{server}/corpus/{index}')
    article = res.json()

    return render_template('detail.html', article=article)

if __name__=="__main__":
    app.run(debug=True, port=2404, host='0.0.0.0')

