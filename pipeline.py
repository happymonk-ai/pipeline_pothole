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
from pymongo import MongoClient
import base64
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
pothole_countlist = []
fram_list = []
total_fram_list =[]
#pothole_perimeter = []
#class_list = []
# class_list = []
# bounding_boxes_list = []
mongo_client_path ="mongodb+srv://amani:SsWRiZBW9qA7E5pM@cluster0.ygyyhu3.mongodb.net"
mongo = None

count_frame = 0


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
    global count_frame
    '''
    ADD DETECTRON (POTHOLE, MANHOLE, CRACK DETECTION) CODE HERE. NO NEED TO DO BATCHING AT THIS POINT.
    '''
    await asyncio.sleep(0.001)
    # cv2.imwrite("./output/save_image"+str(count_frame)+".jpg",image)
    class_list = []
    potho_count = 0
    color = (255, 0, 0)
    thickness = 2
    bounding_boxes_list = []
    image_1 = image
    image = torch.tensor(image).permute(2, 0, 1)
    image_batch = [{"image": image}]
    res = det_model(image_batch)[0]["instances"].get_fields()
    box_res = res["pred_boxes"]
    class_list = res["pred_classes"].numpy().tolist()
    print(len(class_list),"line 75")
    for index, c in enumerate(class_list):
        print(c,"line 76")
        if c >= 1:
            bounding_boxes_list.append(box_res.tensor[index, :].detach().numpy().tolist())
            image_1 = cv2.rectangle(image_1, (int(box_res.tensor[index, :].detach().numpy().tolist()[0]) , int(box_res.tensor[index, :].detach().numpy().tolist()[1])),(int(box_res.tensor[index, :].detach().numpy().tolist()[2]) ,int(box_res.tensor[index, :].detach().numpy().tolist()[3])), color, thickness)
            # cv2.imwrite("./output/save_image"+str(count)+".jpg",image_1)
            print(c , "line 81 image printed")
            # count += 1
        if c == 2:
            potho_count += 1
    # cv2.imwrite("./output/save_image"+str(count_frame)+".jpg",image_1)
    fram_list.append(image_1)
    count_frame += 1
    print(len(bounding_boxes_list),"line 64 pipelin ")
    return bounding_boxes_list , potho_count , fram_list


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


async def count_frames_manual(video, total_frame_count):
    # initialize the total number of frames read
    frames = []
    currentframe = 0
    while(True):
        print(currentframe,"line 113")
        # reading from frame
        ret,frame = video.read()
        # ret = True
        if currentframe >= total_frame_count:
            print("end of video ")
            break 
        elif ret is False:
            continue
        else:
            frames.append(frame)
        currentframe += 1
    video.release()
    return frames


async def main():
    cls_model =  await make_cls_model()
    det_model =  make_det_model(0.9)
    mongo = MongoClient(mongo_client_path)
    db = mongo["potholes_detection"]
    col = db["map_data"]
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
        timestamp_list,latitude_list,longitude_list = await ddpaiinfo(file_path=file_path_gpx)
        video = cv2.VideoCapture(file_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = await count_frames_manual(video, total_frame_count=total_frame_count)
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

    elif camera_type == "gopro":
        timestamp_list,latitude_list,longitude_list= await goprogenerate(file_path=file_path)
        video = cv2.VideoCapture(file_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = await count_frames_manual(video, total_frame_count=total_frame_count)
        batch_size = int(total_frame_count // fps)
        print(batch_size, "line 173")
        count = 0 
        print(len(frames), "line 174")
        for ind , item in enumerate(frames):
            print(ind , "line 177 index")
            isRoad.append(await detect_road(item, cls_model=cls_model))

            if isRoad[-1] == 1:
                print("road detected")
                x , y , z = await detect_pothole(item,det_model=det_model)
                pothole_bounding_box.append(x)
                pothole_countlist.append(y)
                for item_2 in z: 
                    total_fram_list.append(base64.b64encode(cv2.resize(item_2,(512,512))))
            else:
                total_fram_list.append(base64.b64encode(cv2.resize(item,(512,512))))
                pothole_bounding_box.append([])
                pothole_countlist.append(int(0))
                print('No road detected')

            if ind==batch_size:
                begin_pointer = 0
                end_pointer = ind
                batch_bbox_with_ids = await make_unique_ids(pothole_bounding_box[begin_pointer:end_pointer])
                timestamp_list_1 = timestamp_list[begin_pointer:end_pointer]
                longitude_list_1 = longitude_list[begin_pointer:end_pointer]
                latitude_list_1 = latitude_list[begin_pointer:end_pointer]
                pothole_countlist_1 = sum(pothole_countlist[begin_pointer:end_pointer])
                total_fram_list_1 = total_fram_list[begin_pointer:end_pointer]
                final_json_object = await prepare_frame_data(frames, batch_bbox_with_ids,timestamp_list=timestamp_list_1,longitude_list=longitude_list_1,latitude_list=latitude_list_1)
                video_json_object = {"video_id": file_path.split("/")[-1].split(".")[0], "potholes_per_unit_dim": pothole_countlist_1, "frame":total_fram_list_1, "results": final_json_object}
                col.insert_one(video_json_object)
                print(video_json_object, "final 209")
            elif ind % batch_size == 0:
                begin_pointer = ind - batch_size
                end_pointer = ind
                batch_bbox_with_ids = await make_unique_ids(pothole_bounding_box[begin_pointer:end_pointer])
                timestamp_list_1 = timestamp_list[begin_pointer:end_pointer]
                longitude_list_1 = longitude_list[begin_pointer:end_pointer]
                latitude_list_1 = latitude_list[begin_pointer:end_pointer]
                total_fram_list_1 = total_fram_list[begin_pointer:end_pointer]
                pothole_countlist_1 = sum(pothole_countlist[begin_pointer:end_pointer])
                final_json_object = await prepare_frame_data(frames, batch_bbox_with_ids,timestamp_list=timestamp_list_1,longitude_list=longitude_list_1,latitude_list=latitude_list_1)
                video_json_object = {"video_id": file_path.split("/")[-1].split(".")[0], "potholes_per_unit_dim": pothole_countlist_1, "frames":total_fram_list_1, "results": final_json_object}
                col.insert_one(video_json_object)
                print(video_json_object, "final 220")
            count += 1
        
        
    #     for item in frames:
    #         isRoad.append(await detect_road(item, cls_model=cls_model))
    #         if isRoad[-1] == 1:
    #             print("road detected")
    #             x , y = await detect_pothole(item,det_model=det_model)
    #             pothole_bounding_box.append(x)
    #             pothole_countlist.append(y)
    #         else:
    #             pothole_size.append([])
    #             print('No road detected')
    #         count += 1

    # batch_bbox_with_ids = await make_unique_ids(pothole_bounding_box)
    # batch_size = int(total_frame_count // fps)
    # print(batch_size,"line170" , total_frame_count, fps)
    # print(len(pothole_countlist), "line 181")
    # print(len(batch_bbox_with_ids),len(timestamp_list),len(longitude_list),len(latitude_list),len(pothole_bounding_box),"batch box list line 187")
    # for ind , item in enumerate(pothole_countlist):
    #     batch_bbox_with_ids_1 =batch_bbox_with_ids[ind:ind+batch_size]
    #     timestamp_list_1 = timestamp_list[ind:ind+batch_size]
    #     longitude_list_1 = longitude_list[ind:ind+batch_size]
    #     latitude_list_1 = latitude_list[ind:ind+batch_size]
    #     final_json_object = await prepare_frame_data(frames, batch_bbox_with_ids_1,timestamp_list=timestamp_list_1,longitude_list=longitude_list_1,latitude_list=latitude_list_1)
    #     video_json_object = {"video_id": file_path.split("/")[-1].split(".")[0], "results": final_json_object}
    #     col.insert_one(video_json_object)
    #     print(final_json_object)
    #     ind+=batch_size

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