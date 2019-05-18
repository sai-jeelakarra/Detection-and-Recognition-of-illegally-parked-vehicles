
import cv2
import text_recognition





def detect_plate(image):


    image_name = image

    img = cv2.imread(image)
    image_name=image_name[:-4]


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    contours,h = cv2.findContours(thresh,1,2)


    largest_rectangle = [0,0]
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        if len(approx)==4:
            area = cv2.contourArea(cnt)
            if area > largest_rectangle[0]:

                largest_rectangle = [cv2.contourArea(cnt), cnt, approx]

    x,y,w,h = cv2.boundingRect(largest_rectangle[1])

    roi=img[y:y+h,x:x+w]


    newimage = image_name+"_plate.jpg"

    cv2.imwrite(newimage,roi)

    cv2.imshow(newimage,roi)

    x=text_recognition.recognize(newimage)
    return 0

