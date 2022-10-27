from time import sleep
import torch
import pandas as pd
import pickle
from detectron2.modeling import build_model
import os
import numpy as np
import math
from typing import List, Any, Dict
from calendar import c
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.config import get_cfg
from detectron2 import model_zoo
import random,cv2,matplotlib.pyplot as plt
import asyncio
from pymongo import MongoClient
import os
from gpx_converter import Converter
import codecs
import subprocess
# from road_cls_model import AModel

datalist = {}

def plot_samples(dataset_name,number_of_images_to_plot=1):
    dataset_custom=DatasetCatalog.get(dataset_name)
    dataset_custom_metadata=MetadataCatalog.get(dataset_name)

    for d in random.sample(dataset_custom,number_of_images_to_plot):
        img=cv2.imread(d["file_name"])
        visualizer=Visualizer(img[:,:,::-1],metadata=dataset_custom_metadata,scale=0.5)
        vis=visualizer.draw_dataset_dict(d)
        plt.figure(figsize=(10,10))
        plt.imshow(vis.get_image())
        plt.show()

def get_train_cfg(config_file_path,checkpoint_url,train_dataset_name,test_dataset_name,num_classes,device,output_dir):
    cfg=get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.DATASETS.TRAIN=(train_dataset_name,)
    cfg.DATASETS.TEST=(test_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS=2
    cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.SOLVER.IMS_PER_BATCH=2
    cfg.SOLVER.BASE_LR=0.00025
    cfg.SOLVER.MAX_ITER=1500
    cfg.SOLVER.STEPS=[]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE=128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES=num_classes
    cfg.MODEL.DEVICE=device
    cfg.OUTPUT_DIR=output_dir
    return cfg

def on_image(image_path,predictor):
    im=cv2.imread(image_path)
    outputs=predictor(im)
    v=Visualizer(im[:,:,::-1],metadata={},scale=0.5,instance_mode=ColorMode.SEGMENTATION)
    v=v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #draw labels on image
    plt.figure(figsize=(8,8))
    plt.imshow(v.get_image())
    plt.show()

def on_video(video_path,predictor):
    cap=cv2.VideoCapture(video_path)
    if cap.isOpened()==False:
        print("Error opening video stream or file")
        return
    (success,image)=cap.read()
    while success:
        predictions=predictor(image)
        v=Visualizer(image[:,:,::-1],metadata={},scale=0.5,instance_mode=ColorMode.SEGMENTATION)
        output = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        cv2.imshow("Result",output.get_image()[:,:,::-1])
        if cv2.waitKey(1) & 0xFF==ord("q"):
            break
        (success,image)=cap.read()


# for counting unique id
base_index=0


# cls model class
class AModel(torch.nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()
        self.nC = num_classes
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
        for params in self.backbone.parameters():
            params.requires_grad=False
        self.op_layer = torch.nn.Linear(1000, self.nC)

    def forward(self, x):
        return self.op_layer(self.backbone(x))


def make_det_model(THRESH):
    cfg_save_path = "./weights/OD_cfg.pkl"
    with open(cfg_save_path, "rb") as f:
        cfg = pickle.load(f)
        # path of the pretrained weight
        cfg.MODEL.WEIGHTS = os.path.join(
            cfg.OUTPUT_DIR, "./weights/model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESH
        # define the model
        model = build_model(cfg)
        # using the model for inference
        model.train(False)
        # await asyncio.sleep(0.001)
        return model

async def make_cls_model():
    # classification model
    base_model = AModel(num_classes=2)
    model_path = "./weights/best_model.pt"
    chkp = torch.load(model_path, map_location=torch.device("cpu"))
    # loading pretrained weights
    base_model.load_state_dict(chkp["model"])
    await asyncio.sleep(0.001)
    return base_model

async def get_database():
   # Provide the mongodb atlas url to connect python to mongodb using pymongo
   CONNECTION_STRING = "mongodb+srv://amani:SsWRiZBW9qA7E5pM@cluster0.ygyyhu3.mongodb.net/potholes_detection"
   await asyncio.sleep(0.001)
   # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
   client =  MongoClient(CONNECTION_STRING)
   db =client["potholes_detection"]
   col = db["videos"]
   x = col.find()
   count = 0
   for data in x:
       datalist[count]= data
       count += 1
   for i in datalist.keys():
    camera_type = datalist[i]["cameraType"]
    if camera_type == "gopro":
        file_path = os.path.join(str(datalist[i]["destination"]),str(datalist[i]["name"]))
        return camera_type , file_path
    elif camera_type == "ddpai_x2pro":
        file_path = os.path.join(str(datalist[i]["destination"]),str(datalist[i]["name"])) # ddpai location file
        file_path_gpx = os.path.join(str(datalist[i]["destination"]),str(datalist[i]["name"]))
        return camera_type , file_path, file_path_gpx 
    


# def spatial_density(lat1,lat2,lon1,lon2,num_pothole):
#     lat1 = radians(lat1)#initial value of latitude
#     lat2 = radians(lat2)#Final value of latitude
#     lon1 = radians(lon1)#Initial value of longitude
#     lon2 = radians(lon2)#Final value of longitude
#     del_lat = lat2-lat1
#     del_lon = lon1-lon2
#     a = sin(del_lat/2)**2 + (cos(lat1)*cos(lat2))*(sin(del_lon/2)**2)
#     c = 2*asin(sqrt(a))
#     length = (c*6371*1000) #Radius of Earth = 6371 km
#     print(length)
#     breadth = 4 #estimated width of single lane in meters
#     Density = num_pothole / (length * breadth)
#     return Density

async def calc_pot_area(bbox):
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    await asyncio.sleep(0.001)
    return ((xmax-xmin)*(ymax-ymin)*math.pi/4)


async def calc_severity(bbox, scr_area,  THRESHOLD_LOW=10.0, THRESHOLD_HIGH=20.0):
    pot_area = await calc_pot_area(bbox)
    res=(pot_area / scr_area) * 100
    if res < THRESHOLD_LOW:
        await asyncio.sleep(0.001)
        return "LOW", 2
    elif res<THRESHOLD_HIGH and res>THRESHOLD_LOW:
        await asyncio.sleep(0.001)
        return "MEDIUM", 3
    else:
        await asyncio.sleep(0.001)
        return "HIGH", 5


async def euclidean_distance(bbox1: List[float], bbox2: List[float]) -> bool:
    x1,y1,x2,y2 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x3,y3,x4,y4 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    x_mid_1, y_mid_1 = (x2-x1)/2, (y2-y1)/2
    x_mid_2, y_mid_2 = (x4-x3)/2, (y4-y3)/2
    euclidean_dist = (x_mid_1 - x_mid_2)**2 + (y_mid_1 - y_mid_2)**2
    await asyncio.sleep(0.001)
    return np.sqrt(euclidean_dist)

async def fetch_unique_ids_for_bboxes(frame1_bboxes: Any, frame2_bboxes: List[List[float]], thresh=100): # frame1_bboex: List[Dict[int, List[float]]]
    global base_index
    id_bbox_list = []
    if not frame1_bboxes:
        for bbox in frame2_bboxes:
            id_bbox_list.append({base_index : bbox})
            base_index += 1
    else:
        used_bboxes = set()
        for bbox1 in frame2_bboxes:
            euc_dist_list = []
            for bbox2 in frame1_bboxes:
                _, bbox2_coords = next(iter(bbox2.items()))
                euc_dist_list.append(await euclidean_distance(bbox1, bbox2_coords))
            min_dist = np.inf
            idx = -1
            for index, dist in enumerate(euc_dist_list):
                if min_dist > dist and dist < thresh and index not in used_bboxes:
                    min_dist = dist
                    idx = index
            if idx == -1:
                id_bbox_list.append({base_index : bbox1})
                base_index += 1
            else:
                used_bboxes.add(idx)
                bbox_id, _ = next(iter(frame1_bboxes[idx].items()))
                id_bbox_list.append({bbox_id: bbox1})
    return id_bbox_list

async def make_unique_ids(bbox_list, thresh= 100.0):
    output_list = []
    if len(bbox_list) == 1:
        output_list.append(await fetch_unique_ids_for_bboxes(None, bbox_list[0], thresh=thresh))
    elif len(bbox_list) > 1:
        for index in range(len(bbox_list)):
            if index == 0:
                output_list.append(await fetch_unique_ids_for_bboxes(None, bbox_list[index], thresh=thresh))
            else:
                output_list.append(await fetch_unique_ids_for_bboxes(output_list[-1], bbox_list[index], thresh=thresh))
    return output_list


async def fetch_best_frame_details(bbox_list_with_ids):
    df = pd.DataFrame(columns=["frame_id", "pothole_count", "severity"])
    for frame_index, frame_bboxes in enumerate(bbox_list_with_ids):
        area = 0
        severity = 0
        for bboxes in frame_bboxes:
            for _, box in bboxes.items():
                severity += await calc_severity(box, 1000*2000)[1]
        dict_obj = {
                "frame_id": frame_index,
                "pothole_count": len(frame_bboxes),
                "severity": severity
                }
        df = df.append(dict_obj, ignore_index = True)
    df.sort_values(by=["severity", "pothole_count"], ascending=False, inplace=True)
    df.to_csv("bbox_updated.csv", index=False)
    best_frame = df.loc[0, :].to_dict()
    print(best_frame)
    return best_frame

async def goprogenerate(file_path):
    timestamp_list = []
    latitude_list = []
    longitude_list = []
    if file_path.endswith("mp4") or file_path.endswith("MP4"):
        full_gpx_output_path = file_path.replace(".MP4", ".GPX")
        with open(full_gpx_output_path, "w") as gpx_file : 
            exiftool_command = ["/home/wajoud/pothole_pipeline/Image-ExifTool-12.49/exiftool", "-ee", "-p", "/home/wajoud/pothole_pipeline/weights/gpx.fmt.txt", file_path]
                                ## Exiftool.exe full location                                                         ## Gpx.fmt.txt full location          
            subprocess.run(exiftool_command, stdout=gpx_file)
            gpx_infor = Converter(input_file=full_gpx_output_path).gpx_to_dataframe()
        time=gpx_infor.time
        latitude= gpx_infor.latitude
        longitude= gpx_infor.longitude
        for i in range(len(gpx_infor)):
            timestamp_list.append((str(time[i]).split("+")[0]))
            latitude_list.append(latitude[i])
            longitude_list.append(longitude[i])
    return timestamp_list,latitude_list ,longitude_list

async def ddpaiinfo(file_path):
    timestamp_list = []
    latitude_list = []
    longitude_list = []
    if file_path.endswith("gpx") or file_path.endswith("GPX"):
        count=0
        with codecs.open(file_path, 'r', encoding='utf-8',errors='ignore') as file1:
            for line in file1.readlines():
                if (line.startswith('$GPRMC')):
                    count += 1
                    latiTude = line[len("$GPRMC,"):].split(',')[2]
                    print('latiTude:', latiTude)
                    longiTude = line[len("$GPRMC,"):].split(',')[4]
                    print('longiTude:', longiTude)
                    dateStamp = line[len("$GPRMC,"):].split(',')[8]
                    # print('dateStamp:', dateStamp)
                    timeStamp = line[len("$GPRMC,"):].split(',')[0]
                    dateStamp=dateStamp[0]+dateStamp[1]+"-"+dateStamp[2]+dateStamp[3]+"-20"+dateStamp[4]+dateStamp[5]+" "
                    timeStamp=timeStamp[0]+timeStamp[1]+":"+timeStamp[2]+timeStamp[3]+":"+timeStamp[4]+timeStamp[5:]
                    finaltimestamp=dateStamp+timeStamp
                    print('timeStamp:', finaltimestamp)
                    timestamp_list.append(finaltimestamp)
                    latitude_list.append(latiTude)
                    longitude_list.append(longiTude)
                    print(count)  
        # close and save the files
        file1.close()
        return timestamp_list,latitude_list,longitude_list
    
async def bbox_length(bbox):
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    return max(xmax-xmin, ymax-ymin)

async def bbox_width(bbox):
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3] 
    return min(xmax-xmin, ymax-ymin)
    
async def prepare_frame_data(frames,  bbox_list_with_ids,longitude_list,latitude_list,timestamp_list):
    output_list = []
    for frame, (frame_index, frame_bbox) in zip(frames, enumerate(bbox_list_with_ids)):
        frame_dict = {}
        frame_dict["id"] = frame_index
        #frame_dict["image"] = frame
        frame_dict["geo"] = {"lat":latitude_list , "lon":longitude_list}
        frame_dict["timestamp"] = timestamp_list
        frame_dict["potholes"] = []
        for box in frame_bbox:
            box_id, box_coord = next(iter(box.items()))
            box_dict ={}
            box_dict["id"] = box_id
            box_dict["length"] = await bbox_length(box_coord)
            box_dict["width"] = await bbox_width(box_coord)
            box_dict["surfaceArea"] = await calc_pot_area(box_coord)
            box_dict["perimeter"] = (box_dict["length"] + box_dict["width"])*2
            box_dict["severity"] = await calc_severity(box_coord, 1000*2000)
            frame_dict["potholes"].append(box_dict)
        output_list.append(frame_dict)
    return output_list


def generate_json_object(frames, bbox_list_with_ids):
        pass