from flask import Flask, request, redirect, render_template, Markup
app = Flask(__name__)
# app._static_folder = "/home/vijay/hackathon/lahacks/uber_squats"

import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
import subprocess
@app.route('/challenge', methods = ['POST'])
def challenge():
        username = request.form['username']
        challenge_type = request.form['challenge_type']
        print("Challenge Type = "+challenge_type)

        subprocess.run(["python","run_webcam.py", "--model=mobilenet_thin" ,"--resize=432x368","--camera=0"])
        return render_template('end.html', username_ret = username)


@app.route('/qrcode', methods = ['POST'])
def qrcode():
	cap = cv2.VideoCapture(0)
	font = cv2.FONT_HERSHEY_PLAIN
	qr_scanned = False
	username = ""

	while True:
		_, frame = cap.read()

		decodedObjects = pyzbar.decode(frame)
		for obj in decodedObjects:
			#print("Data", obj.data)
			cv2.putText(frame, str(obj.data), (50, 50), font, 2, (255, 0, 0), 3)
			# print(str(obj.data.decode("utf-8")))
			username = str(obj.data.decode("utf-8"))
			qr_scanned = True
			break;

		cv2.imshow("Frame", frame)

		if qr_scanned:
			cap.release()
			cv2.destroyAllWindows()
			break;

		key = cv2.waitKey(1)
		if key == 27:
			break
	
	return render_template('challenge.html', username_ret = username) 

@app.route('/')
def home():
	return render_template('index.html')

if __name__ == '__main__':
	app.run()

print ('')


