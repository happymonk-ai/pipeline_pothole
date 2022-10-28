import asyncio
import gc
import os
import cv2
import json
import torch
import pickle
from utils import *
from glob import glob
from PIL import Image
from nanoid import generate
from detectron2.modeling import build_model
from torchvision.transforms import ToTensor, Resize, Normalize, CenterCrop, Compose
from datetime import date
from gpx_converter import Converter
import codecs
import subprocess
import pandas as pd
#dummy_geo_lat = 5555.55
#dummy_geo_long = 9999.99
# path to single video input, need to put this in loop if multiple videos are to be read automatically

path = './Testvid.mp4'
isRoad = []
fps=60
avg_Batchcount_pothole = []
avg_Batchcount_manhole = []
avg_Batchcount_crack = []
pothole_size = []
pothole_spatial_density = []
pothole_severity = []
timestamps_video=[]
timestamps_gps=[]
latlong=[]
pothole_info = []
pothole_bounding_box = []
#pothole_perimeter = []
#class_list = []
# class_list = []
# bounding_boxes_list = []

base_augmentation = Compose(
    [
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ]
)


async def detect_pothole(image, det_model):
    '''
    ADD DETECTRON (POTHOLE, MANHOLE, CRACK DETECTION) CODE HERE. NO NEED TO DO BATCHING AT THIS POINT.
    '''
    await asyncio.sleep(0.001)
    count = 0
    class_list = []
    color = (255, 0, 0)
    thickness = 2
    bounding_boxes_list = []
    image_1 = image
    image = torch.tensor(image).permute(2, 0, 1)
    image_batch = [{"image": image}]
    res = det_model(image_batch)[0]["instances"].get_fields()
    box_res = res["pred_boxes"]
    class_list = res["pred_classes"].numpy().tolist()
    for index, c in enumerate(class_list):
        if c >= 1:
            bounding_boxes_list.append(box_res.tensor[index, :].detach().numpy().tolist())
            image_1 = cv2.rectangle(image_1, (int(box_res.tensor[index, :].detach().numpy().tolist()[0]) , int(box_res.tensor[index, :].detach().numpy().tolist()[1])),(int(box_res.tensor[index, :].detach().numpy().tolist()[2]) ,int(box_res.tensor[index, :].detach().numpy().tolist()[3])), color, thickness)
            cv2.imwrite("./output/save_image"+str(count)+".jpg",image_1)
            count += 1
    print(len(bounding_boxes_list),"line 64 pipelin ")
    return bounding_boxes_list  


async def detect_road(image,cls_model):
    '''
    ADD ROAD DETECTION CODE HERE. NO NEED TO DO BATCHING AT THIS POINT. OUTPUT SHOULD BE 0 OR 1
    '''
    cls_frame = Image.fromarray(image)
    cls_frame = base_augmentation(cls_frame).unsqueeze(0)
    isRoad = torch.argmax(cls_model(cls_frame),
                          dim=1).detach().numpy().tolist()[0]
    # The value (0 or 1) should come from the result of road detection module above this line
    print(isRoad,"line 78 pipeline")
    return isRoad


async def count_frames_manual(video):
    # initialize the total number of frames read
    total = 0
    frames = []
    # loop over the frames of the video
    while True:
        (ret, frame) = video.read()
        timestamps_video.append(video.get(cv2.CAP_PROP_POS_MSEC))
        latlong.append([np.random.randint(-180,180),np.random.randint(0,360)])
        if not ret:
            break
        total += 1
        frames.append(frame)
    return len(frames), frames


async def main():
    cls_model =  await make_cls_model()
    det_model =  make_det_model(0.7)
    desc = await get_database()
    tuple_length = len(desc)
    if tuple_length == 2:
        camera_type = desc[0]
        file_path = desc[1]
    else:
        camera_type = desc[0]
        file_path = desc[1]
        file_path_gpx =desc[2]
    print(camera_type, file_path, "mongo db")
    # camera_type = "gopro"
    # file_path = "./GH052755.MP4"
    if camera_type == "ddpai_x2pro":
        print("i am ddpai_x2pro")
        timestamp_list,latitude_list,longitude_list = await ddpaiinfo(file_path=file_path_gpx)
        video = cv2.VideoCapture(file_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frame_count, frames = await count_frames_manual(video)
        count = 0 
        for item in frames:
            isRoad.append(await detect_road(item, cls_model=cls_model))
            if isRoad[-1] == 1:
                print("road detected")
                pothole_bounding_box.append(await detect_pothole(item,det_model=det_model))
            else:
                pothole_size.append([])
                print('No road detected')
            count += 1
            if count > 10 :
                break
            
    elif camera_type == "gopro":
        print("i am goprob")
        timestamp_list,latitude_list,longitude_list= await goprogenerate(file_path=file_path)
        video = cv2.VideoCapture(file_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frame_count, frames = await count_frames_manual(video)
        count = 0 
        for item in frames:
            isRoad.append(await detect_road(item, cls_model=cls_model))
            if isRoad[-1] == 1:
                print("road detected")
                pothole_bounding_box.append(await detect_pothole(item,det_model=det_model))
            else:
                pothole_size.append([])
                print('No road detected')
            count += 1
            if count > 10 :
                break

    batch_bbox_with_ids = await make_unique_ids(pothole_bounding_box)
    final_json_object = await prepare_frame_data(frames, batch_bbox_with_ids,timestamp_list=timestamp_list[-1],longitude_list=longitude_list[-1],latitude_list=latitude_list[-1])
    print(final_json_object)

    '''
    ADD ALL THE BATCH LOGIC HERE TO COMPUTE SPATIAL DENSITY AND SEVERITY
    '''
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try :
        loop.run_until_complete(main())
    except RuntimeError as e:
        print("error ", e)
        print(torch.cuda.memory_summary(device=None, abbreviated=False), "cuda")
        
        
'''
# JSON output format
metapothole ={
            "count": '3',
            "length": '[5, 3, 4]',
            "width": '[2, 5, 5]',
            "perimeter": '[14, 16, 18]',
            "surface area":'[10, 15, 20]',
            "severity": '0' # 0: Low, 1:Medium, 2: High 
            }
    

metaObj = {
            "pothole":metapothole
            }

metaBatch = {
    "batchid": '0',
    "road_detect": '1', # 1: Detected, 0: Not detected
    "object": metaObj}

primary = { "videoId":'XYZ.MP4',
            "frameId":'0', 
            "timestamp":'2022-09-21 13:51:49.845000', 
            "geo":{
                'lat':'17.4495748'
                'lon':'78.3826457'
            },
            "metaData": metaBatch}
'''