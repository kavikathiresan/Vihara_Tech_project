import cv2
import numpy as np
import argparse
import sys
import redis
import  logging
logging.basicConfig(format='%(filename)s:%(message)s',level=logging.DEBUG)

# Connect to Redis
redis_client = redis.StrictRedis(host='localhost', port=6379, decode_responses=True)
class Yolov3:
    def __init__(self):
        self.weights = op.weights # weight file
        self.cfg = op.cfg     # cfg file
        self.Threshold = 0.5
        self.image_size = op.image_size
        self.neural_networks = cv2.dnn.readNetFromDarknet(self.cfg,self.weights)
        self.output = self.neural_networks.getUnconnectedOutLayersNames()

    def Bounding_box(self,detections):
        try:
            """ Marking the coordinates of x,y,w,h based on image of probability """
            coordinates = []
            class_label = []
            confidence_score = []

            for i in detections:
                for j in i:
                    probability = j[5:]
                    class_index = np.argmax(probability)
                    confidence = probability[class_index]

                    if confidence > self.Threshold:
                        w,h = int(j[2] * self.image_size), int(j[3] * self.image_size)
                        x,y = int(j[0] * self.image_size -w/2),int(j[1] * self.image_size -h/2)
                        coordinates.append([x,y,w,h])
                        class_label.append(class_index)
                        confidence_score.append(confidence)

            final_box = cv2.dnn.NMSBoxes(coordinates,confidence_score,self.Threshold,.6)
            return final_box,coordinates,confidence_score,class_label

        except Exception as e:
            exec_type = sys.exc_info()
            logging.error(f'error type:{exec_type} & error in main:{e.__str__()}')

    def Prediction_box(self,final_box,coordinates,confidence_score,class_label,width_ratio,height_ratio,video):
        try:
            class_name = []
            """ Taking classes names """
            k = open('class_names', 'r')
            for i in k.readlines():
                class_name.append(i.strip())
            """ Capturing the object detection frame and marking with rectangle box and giving class name and their probability score """
            Total_person_detects = []
            count_1 = 0
            if len(final_box)>0:
                for j in final_box.flatten():
                    if class_name[class_label[j]]=='person':
                        count_1 +=1
                        Total_person_detects.append(count_1)
                        x, y, w, h = coordinates[j]
                        x = int(x * width_ratio)
                        y = int(y * height_ratio)
                        w = int(w * width_ratio)
                        h = int(h * height_ratio)
                        cnf = str(round(confidence_score[j],2))
                        text = str(class_name[class_label[j]]) + " " + cnf
                        cv2.rectangle(video,(x,y),(x+w,y+h),(255,0,0),1,cv2.LINE_AA)
                        cv2.putText(video,text,(x,y-3),cv2.FONT_HERSHEY_TRIPLEX,1,(0,255,0),1,cv2.LINE_4)
                        """ storing the person coordinates in redis """
                        bounding_box_person={'x':x,'y':y,'w':w,'h':h}
                        redis_client.rpush('person_bounding_boxes', str(bounding_box_person))
            else:
                logging.info(f"No detection ")

        except Exception as e:
            exec_type = sys.exc_info()
            logging.error(f'error type:{exec_type} & error in main:{e.__str__()}')

    """ setting up the video and giving to architecture """
    def Inference(self,video,original_width,original_height):
        try:
            blob = cv2.dnn.blobFromImage(video,1/255,(320,320),True,crop=False)
            self.neural_networks.setInput(blob)
            output_layers =self.neural_networks.forward(self.output)
            final_box,coordinates,confidence_score,class_label=self.Bounding_box(output_layers)
            self.Prediction_box(final_box,coordinates,confidence_score,class_label,original_width/320,original_height/320,video)

        except Exception as e:
            exec_type = sys.exc_info()
            logging.error(f'error type:{exec_type} & error in main:{e.__str__()}')


""" Passing the argument Directly """
if __name__=='__main__':
    parse=argparse.ArgumentParser()
    parse.add_argument('--weights',type=str,default='yolov3.weights',help='weight path')
    parse.add_argument('--cfg',type=str,default='yolov3.cfg',help='cfg path')
    parse.add_argument('--video',type=str,default='',help='video path')
    parse.add_argument('--image_size',type=int,default=320,help='size of w & h')
    op = parse.parse_args()
    obj = Yolov3()

    """ Taking the video and predicting the object detection frame based on the classes  """
    if op.video:
        try:
            cap = cv2.VideoCapture(op.video)
            width = cap.get(3)
            height = cap.get(4)
            fps = cv2.CAP_PROP_FPS
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_file = cv2.VideoWriter('final_output.avi',fourcc,fps,(int(width),int(height)))
            while cap.isOpened():
                res,frame = cap.read()
                if res == True:
                    obj.Inference(video=frame,original_width=width,original_height=height)
                    cv2.imshow('final_output',frame)
                    output_file.write(frame)
                    if cv2.waitKey(3) & 0xFF==ord('q'):
                        break
                else:
                    break
            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            exec_type = sys.exc_info()
            logging.error(f'error type:{exec_type} & error in main:{e.__str__()}')



