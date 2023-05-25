import cv2
import torch
from helpers import get_device,get_classes,load_model,load_video,get_classes,modify_coordinates

class VideoObjectDetection:

    ## 
    def __init__(self):
        self.device=get_device
        self.model=load_model()
        self.classes=get_classes(self.model)
        

    def predict_frame(self,frame):
        frame = [frame]
        output = self.model(frame)
        labels, cord = output.xyxyn[0][:, -1].to("cpu").numpy(), output.xyxyn[0][:, :-1].to("cpu").numpy()
        return labels,cord

    def plot_results(self,frame,labels,cord):
        
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.classes[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame
       


    def process_video(self,path,obj_name):
        video=load_video(path)
        x_size=int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_size=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ext=cv2.VideoWriter_fourcc(*"MJPG")
        out_video=cv2.VideoWriter(obj_name,ext,10,(x_size,y_size))
        while True:
            ret,frame=video.read()
            if ret==True:
                labels,coords=self.predict_frame(frame)
                frame=self.plot_results(frame,labels,coords)
                out_video.write(frame)
                print("running")
            else:
                break
        video.release()
        out_video.release()
 
           
            




    


video_obj=VideoObjectDetection()
video_obj.process_video("2.mp4","2_modified.mp4")