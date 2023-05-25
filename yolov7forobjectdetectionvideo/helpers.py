import torch
import cv2 as cv


#return the device
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"



#this script is used to export the model
def load_model(weight_file_path="yolov7.pt"):
    model=torch.hub.load("WongKinYiu/yolov7","custom",weight_file_path,trust_repo=True)
    model.to(get_device())
    #put model in evaluation mode
    model.eval()
    return model


#get the video from the path
def load_video(path):
    video=cv.VideoCapture(path)
    return video


##return the class list for the model
def get_classes(model):
    return model.names

def modify_coordinates(bboxes,old_shape,new_shape):
    mul_factx=old_shape[0]/new_shape[0]
    mul_facty=old_shape[1]/new_shape[1]
    for i in range(len(bboxes)):
        bboxes[i][0],bboxes[i][2],bboxes[i][1],bboxes[i][3]=int(bboxes[i][0]*mul_factx),int(bboxes[i][2]*mul_factx),int(bboxes[i][1]*mul_facty),int(bboxes[i][3]*mul_facty)
    return bboxes

