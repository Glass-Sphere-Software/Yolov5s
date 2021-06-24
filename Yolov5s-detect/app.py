from contextvars import Token

import requests

import requests
from flask import Flask
from flask import request
from flask import jsonify
from authlib.oauth2.rfc6750 import BearerTokenValidator
from authlib.oauth2.rfc7662 import IntrospectTokenValidator
from authlib.integrations.flask_oauth2 import ResourceProtector, current_token, token_authenticated

import argparse
import io
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from minio import Minio

from models.experimental import attempt_load, attempt_loadFromBytes
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from datetime import datetime, timezone

from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport

userInfoURL = "https://ens-fiti.de/user/info"

import asyncio
from functools import wraps
from client.gsdbs import GSDBS
import pandas as pd



class MyIntrospectTokenValidator(IntrospectTokenValidator):
    def introspect_token(self, token_string):
        headers = {'Authorization': 'Bearer ' + token_string}
        try:
            resp = requests.get(userInfoURL, headers=headers)
            resp.raise_for_status()
            userinfo = resp.json()
            userinfo['token_string'] = token_string
            userinfo['active'] = True
            return userinfo
        except:
            userinfo = {'active': False}
            return userinfo


require_oauth = ResourceProtector()
require_oauth.register_token_validator(MyIntrospectTokenValidator())

app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
@require_oauth()
def analyseVideo():
    if (current_token.get("mandant") is None):
        return jsonify("error token: mandant")
    if (current_token.get("username") is None):
        return jsonify("error token username")

    data = request.get_json()
    if (data.get('classifier') is None):
        return jsonify("error parameter classifier")


    if (data.get('source') is None):
        return jsonify("error parameter source")
    if (data.get('modelname') is None):
        return jsonify("error parameter modelname")
    if (data.get('device') is None):
        return jsonify("error parameter device")
    if (data.get('view_img') is None):
        return jsonify("error parameter view_img")
    if (data.get('save_txt') is None):
        return jsonify("error parameter save_txt")
    if (data.get('imgsz') is None):
        return jsonify("error parameter imgsz")
    if (data.get('nosave') is None):
        return jsonify("error parameter nosave")
    if (data.get('conf_thres') is None):
        return jsonify("error parameter conf_thres")
    if (data.get('iou_thres') is None):
        return jsonify("error parameter iou_thres")
    if (data.get('save_conf') is None):
        return jsonify("error parameter save_conf")

    detect(current_token.get('mandant'),
           current_token.get('username'),
           data.get('classifier'),
           data.get('streamkey'),
           data.get('fragmentid'),
           data.get('source'),
           data.get('modelname'),
           data.get('device'),
           data.get('view_img'),
           data.get('save_txt'),
           data.get('imgsz'),
           data.get('nosave'),
           data.get('conf_thres'),
           data.get('iou_thres'),
           data.get('save_conf'),
           )
    return jsonify("success")


def detect(mandant, username, classifier, streamkey, fragmentid, source, modelname, device, view_img, save_txt, _imgsz,
           _nosave, conf_thres,
           iou_thres, save_conf):
    imgsz = int(_imgsz)
    conf_thres = float(conf_thres)
    iou_thres = float(iou_thres)
    nosave = not bool(_nosave)

    project = 'runs/detect'
    name = 'exp'
    exist_ok = False
    augment = False
    classes = None
    agnostic = False

    _gsdbs = GSDBS()
    df = pd.DataFrame({
        'streamkey': pd.Series([], dtype='str'),
        'fragmentid': pd.Series([], dtype='str'),
        'frameid': pd.Series([], dtype='str'),
        'objectid': pd.Series([], dtype='str'),
        'classid': pd.Series([], dtype='str'),
        'x': pd.Series([], dtype='float'),
        'y': pd.Series([], dtype='float'),
        'width': pd.Series([], dtype='float'),
        'height': pd.Series([], dtype='float'),
        'confidence': pd.Series([], dtype='float')
    })

    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    model = attempt_load(modelname, map_location=device)     # load FP32 model

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        # model.half().to('cuda')
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    t4 = datetime.now()
    # print ('Time for connectong to videostream : ', str(t4-t3))

    frameID = -1
    for path, img, im0s, vid_cap in dataset:
        frameID += 1
        # print("serial frameID: ", frameID, end=", " )
        if (frameID+1) % 1 != 0:
            continue
        # print("selected frameID: ", frameID)

        if (np.all(im0s[0] == 0)):
            break

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                objectid = 0

                for *xyxy, conf, cls in reversed(det):
                    objectid += 1
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (frameID, cls, *xywh, conf) if save_conf else (frameID, cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        # print(cls)
                        # print(conf.data.cpu().numpy())

                        # print(cls)
                        # print(cls.item())

                        classID = str(cls.item()).split(".")[0]
                        row = {'streamkey': streamkey,
                               'fragmentid': fragmentid,
                               'frameid': frameID,
                               'objectid': objectid,
                               'classid': classID,
                               'x': xywh[0],
                               'y': xywh[1],
                               'width': xywh[2],
                               'height': xywh[3],
                               'confidence': conf.data.cpu().numpy()
                               }
                        df = df.append(row, ignore_index=True)

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

    # write to GS-DBS
    rc = _gsdbs.addDObject("coco", ["streamkey", "fragmentid", "frameid", "objectid"], df, schemaCheck=False)
    print(rc)

    return


if __name__ == '__main__':
    app.run()
