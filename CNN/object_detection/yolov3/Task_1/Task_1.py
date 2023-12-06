import cv2
import numpy as np
import argparse
import sys
import logging
logging.basicConfig(level=logging.DEBUG,format='%(filename)s:%(message)s')
class Yolo_v3:
    def __init__(self):
        self.weights = op.weights # weight file
        self.cfg = op.cfg  # cfg file
        self.neural_networks = cv2.dnn.readNetFromDarknet(self.cfg,self.weights)
        self.output=self.neural_networks.getUnconnectedOutLayersNames()
        self.image_size = op.image_size

    def bounding_box(self,detections):
        try:
            Threshold = 0.5
            coordinates = []
            confidence_score=[]
            class_label=[]
            """ Marking the coordinates of x,y,w,h based on image of probability """

            for i in detections:
                for j in i:
                    probability = j[5:] # x,y,w,h,prob
                    class_index = np.argmax(probability)
                    confidence = probability[class_index]

                    if confidence > Threshold:
                        w,h = int(j[2] * self.image_size), int(j[3] * self.image_size)
                        x,y = int(j[0] * self.image_size - w/2), int(j[1] * self.image_size - h/2)
                        coordinates.append([x,y,w,h])
                        class_label.append(class_index)
                        confidence_score.append(confidence)
            final_box = cv2.dnn.NMSBoxes(coordinates,confidence_score,Threshold,.6)
            return final_box,coordinates,confidence_score,class_label

        except Exception as e:
            exc_type = sys.exc_info()
            logging.error(f'error type:{exc_type} and error in main:{e.__str__()}')

    def prediction(self,final_box,coordinates,confidence_score,class_label,width_ratio,height_ratio,image):
        try:
           """ taking the class file """
           class_names = []
           k = open('class_names', 'r')
           for j in k.readlines():
               class_names.append(j.strip())
           """ Only detecting 2 classes : car and person """
           for i in final_box.flatten():
                if class_names[class_label[i]] == 'person':
                    x, y, w, h = coordinates[i]
                    x = int(x * width_ratio)
                    y = int(y * height_ratio)
                    w = int(w * width_ratio)
                    h = int(h * height_ratio)
                    class_prob = str(round(confidence_score[i], 2))
                    text = str(class_names[class_label[i]]) + ":" + class_prob
                    cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(original_image, text, (x, y - 9), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 1)

                elif class_names[class_label[i]] == 'car':
                    x, y, w, h = coordinates[i]
                    x = int(x * width_ratio)
                    y = int(y * height_ratio)
                    w = int(w * width_ratio)
                    h = int(h * height_ratio)
                    class_prob = str(round(confidence_score[i], 2))
                    text = str(class_names[class_label[i]]) + ":" + class_prob
                    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2, cv2.LINE_8)
                    cv2.putText(original_image, text, (x, y - 3), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
                else:
                    break

        except Exception as e:
            exc_type = sys.exc_info()
            logging.error(f'error type:{exc_type} and error in main:{e.__str__()}')


    """ Setup the image and give to architecture """
    def Inference_image(self,original_image,original_width,original_height):
        try:
            blob = cv2.dnn.blobFromImage(original_image,1/255,(320,320),True,crop=False)
            self.neural_networks.setInput(blob)
            output_data = self.neural_networks.forward(self.output)
            final_box,coordinates,confidence_score,class_label=self.bounding_box(output_data)
            out_box=self.prediction(final_box, coordinates, confidence_score, class_label, original_width/320, original_height/320,original_image)
            return out_box

        except Exception as e:
            exc_type = sys.exc_info()
            logging.error(f'error type:{exc_type} and error in main:{e.__str__()}')


if __name__=='__main__': # passing the arguments directly
    parse = argparse.ArgumentParser()
    parse.add_argument('--weights',type=str,default='yolov3.weights',help='weight path')
    parse.add_argument('--cfg',type=str,default='yolov3.cfg',help='cfg path')
    parse.add_argument('--image',type=str,default='',help='image path')
    parse.add_argument('--image_size', type=int, default=320, help='size of width * height')
    op = parse.parse_args()

    obj = Yolo_v3() 

    if op.image:
        try:
            original_image = cv2.imread(op.image,1)
            original_width ,original_height = original_image.shape[1] ,original_image.shape[0]
            obj.Inference_image(original_image=original_image, original_width=original_width, original_height=original_height)
            cv2.imshow('origin',original_image)
            cv2.imwrite('origin.jpg', original_image) # save the output image
            cv2.waitKey()
            cv2.destroyAllWindows()
        except Exception as e:
            exc_type= sys.exc_info()
            logging.error(f'error type:{exc_type} and error in main:{e.__str__()}')




