
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import cv2
import csv

import datetime
from dateutil import parser

mintime = 10


def decode_predictions(scores, geometry):

	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(0, numRows):

		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]


		for x in range(0, numCols):

			if scoresData[x] < 0.5:
				continue


			(offsetX, offsetY) = (x * 4.0, y * 4.0)


			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)


			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]


			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)


			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])


	return (rects, confidences)





def recognize(imagename):




	east =   "frozen_east_text_detection.pb"

	image = cv2.imread(imagename)
	orig = image.copy()
	(origH, origW) = image.shape[:2]


	(newW, newH) = (320, 320)
	rW = origW / float(newW)
	rH = origH / float(newH)


	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]


	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]


	print("\n\nLoading text detector...")
	net = cv2.dnn.readNet(east)


	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)


	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)


	results = []


	for (startX, startY, endX, endY) in boxes:

		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		ht = image.shape[0]
		wt = image.shape[1]

		if wt / ht <= 3.5:
			padding = 1.5
		else:
			padding= 0.05






		dX = int((endX - startX) * padding)
		dY = int((endY - startY) * padding)


		startX = max(0, startX - dX)
		startY = max(0, startY - dY)
		endX = min(origW, endX + (dX * 2))
		endY = min(origH, endY + (dY * 2))


		roi = orig[startY:endY, startX:endX]


		config = ("-l eng --oem 1 --psm 7")
		text = pytesseract.image_to_string(roi, config=config)


		results.append(((startX, startY, endX, endY), text))


	results = sorted(results, key=lambda r:r[0][1])


	for ((startX, startY, endX, endY), text) in results:

		print("\n -------------  Dectected Vehicle NUmber  ------------------ \n ")


		text = list(text)

		for i in range(0,len(text)):
			if not (47<ord(text[i])<58 or 64<ord(text[i])<91) :
				text[i]=" "




		text = "".join(text)
		text = text.replace(" ","")

		text = list(text)

		if len(text)>5:
			if text[2]=="O": text[2]="0"
			if text[3]=="O": text[3]="0"
			if text[2]=="I": text[2]="1"
			if text[3]=="I" : text[3]="1"
			if text[2]=="Z": text[2]="2"
			if text[3]=="Z" : text[3]="2"
			if text[2]=="G": text[2]="6"
			if text[3]=="G" : text[3]="6"
			if text[2]=="B": text[2]="8"
			if text[3]=="B" : text[3]="8"

		text = "".join(text)





		print("{}\n".format(text))


		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
		output = orig.copy()
		cv2.rectangle(output, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(output, text, (startX, startY - 20),
			cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)





		doc = []
		with open("collecteddata.csv", "r") as f:
			reader = csv.reader(f, delimiter=',')
			for row in reader:
				doc.append(row)




		end = datetime.datetime.now().replace(microsecond=0)

		endmins = end.hour*60+end.minute



		flag = 1

		for i in range(0, len(doc)):
			if doc[i][2] == text:

				start = parser.parse(doc[i][0])

				startmins = start.hour*60+start.minute

				doc[i][3] = endmins-startmins
				doc[i][1] = end
				flag = 0
				break

		if (flag == 1):
			newrow = [end, end, text, "0"]
			doc.append(newrow)

		with open("collecteddata.csv", "w") as f:
			writer = csv.writer(f)
			writer.writerows(doc)

		print("\n\n-----------------ILLEGALLY PARKED VEHICLES AT NO PARKING AREA ----------------\n")


		for i in range(1,len(doc)):
			if int(doc[i][3])>=mintime:
				print("vehicle number : ",doc[i][2])


		return 0




