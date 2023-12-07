import numpy as np
import cv2
import sys
import argparse
import logging

class Yolov3:
    def __init__(self):
        self.weights = op.weights # weight file
        self.cfg = op.cfg # cfg file
        self.Threshold = 0.5
        self.neural_networks =cv2.dnn.readNetFromDarknet(self.cfg,self.weights)
        self.out_layers = self.neural_networks.getUnconnectedOutLayersNames()
        self.img_size = op.img_size

    def Bounding_box(self,detections):
        try:
            coordinates = []
            class_label = []
            confidence_score = []
            """ Marking the coordinates of x,y,w,h based on image of probability """
            for i in detections:
                for j in i:
                    probability = j[5:]
                    class_index = np.argmax(probability)
                    confidence = probability[class_index]

                    if confidence > self.Threshold:
                        w,h = int(j[2] * self.img_size),int(j[3] * self.img_size)
                        x,y = int(j[0] * self.img_size -w/2),int(j[1] * self.img_size -h/2)
                        coordinates.append([x,y,w,h])
                        class_label.append(class_index)
                        confidence_score.append(confidence)
            final_box = cv2.dnn.NMSBoxes(coordinates,confidence_score,self.Threshold,.6)
            return final_box,coordinates,confidence_score,class_label

        except Exception as e:
            exc_type = sys.exc_info()
            logging.error(f'type of error:{exc_type}  & error in main:{e.__str__()}')

    def Prediction_box(self,final_box,coordinates,confidence_score,class_label,width_ratio,height_ratio,video):
        try:
            """ Taking the class names """
            class_names = []
            k = open('class_names','r')
            for i in k.readlines():
                class_names.append(i.strip())

            """ Only detecting 2 classes : car and person """
            Total_cars_detects = []
            Total_person_detects = []
            count_1 = 0
            count_2 = 0
            if len(final_box)>0:
                for j in final_box.flatten():
                    if class_names[class_label[j]] == 'car':
                        count_1 += 1
                        Total_cars_detects.append(count_1)
                        x,y,w,h = coordinates[j]
                        x = int(x * width_ratio)
                        y = int(y * height_ratio)
                        w = int(w * width_ratio)
                        h = int(h * height_ratio)
                        cnf = str(round(confidence_score[j],2))
                        text = str(class_names[class_label[j]]) + " "+cnf
                        cv2.rectangle(video,(x,y),(x+w,y+h),(255,0,0),1,cv2.LINE_4)
                        cv2.putText(video,text,(x,y-3),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                        text_1 = f'total car detects:{Total_cars_detects}'
                        if len(Total_cars_detects)>0:
                            cv2.putText(video,text_1,(10,25),cv2.FONT_HERSHEY_TRIPLEX,1,(255,0,0),1,cv2.LINE_AA)
                        #cv2.imshow('frames', video)

                    elif class_names[class_label[j]] == 'person':
                        count_2 += 1
                        Total_person_detects.append(count_1)
                        x, y, w, h = coordinates[j]
                        x = int(x * width_ratio)
                        y = int(y * height_ratio)
                        w = int(w * width_ratio)
                        h = int(h * height_ratio)
                        cnf = str(round(confidence_score[j], 2))
                        text = str(class_names[class_label[j]]) + " " + cnf
                        cv2.rectangle(video, (x, y), (x + w, y + h), (255, 0, 0), 1, cv2.LINE_4)
                        cv2.putText(video, text, (x, y - 3), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                        cv2.putText(video, text, (x, y - 3), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                        text_2 = f'total person detects:{Total_person_detects}'
                        if len(Total_person_detects) > 0:
                            cv2.putText(video, text_2, (10,55), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,0,0),1,cv2.LINE_AA)
                        #cv2.imshow('frames', video)

        except Exception as e:
            exc_type = sys.exc_info()
            logging.error(f'type of error:{exc_type}  & error in main:{e.__str__()}')


    """ Setup the video and give to architecture """
    def Inference(self,video,original_width,original_height):
        try:
            blob = cv2.dnn.blobFromImage(video,1/255,(320,320),True,crop=False)
            self.neural_networks.setInput(blob)
            output_data = self.neural_networks.forward(self.out_layers)
            final_box,coordinates,confidence_score,class_label=self.Bounding_box(output_data)
            out_box = self.Prediction_box(final_box,coordinates,confidence_score,class_label,original_width/320,original_height/320,video)
            return out_box

        except Exception as e:
            exc_type = sys.exc_info()
            logging.error(f'type of error:{exc_type}  & error in main:{e.__str__()}')


""" passing the arguments Directly """
if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--weights',type=str,default='yolov3.weights',help='weight path')
    parse.add_argument('--cfg',type=str,default='yolov3.cfg',help='cfg path')
    parse.add_argument('--video',type=str,default='',help='video path')
    parse.add_argument('--img_size',type=int,default=320,help='size of w&h')
    op = parse.parse_args()
    obj=Yolov3()

    if op.video:
        try:
            cap = cv2.VideoCapture(op.video) # capture the video
            width = cap.get(3)
            height = cap.get(4)
            fps = cv2.CAP_PROP_FPS
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_file = cv2.VideoWriter("final_video.avi",fourcc,fps,(int(width),int(height)))
            while cap.isOpened():
                res,frame = cap.read()
                if res == True:
                    obj.Inference(video=frame,original_width=width,original_height=height)
                    cv2.imshow('final_video',frame)
                    output_file.write(frame)
                    if cv2.waitKey(3) & 0xFF ==ord('q'):
                        break
                else:
                    break
            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            exc_type = sys.exc_info()
            logging.error(f'type of error:{exc_type}  & error in main:{e.__str__()}')

