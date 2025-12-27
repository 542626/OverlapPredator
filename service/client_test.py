import json
import requests
import numpy as np
import os
import sys
import torch
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(ROOT_DIR)

# 服务地址
URL = "http://127.0.0.1:8000"

src = torch.load("/home/dl/workspace/python_project/OverlapPredator/data/indoor/test/7-scenes-redkitchen/cloud_bin_0.pth").tolist()
tgt = torch.load("/home/dl/workspace/python_project/OverlapPredator/data/indoor/test/7-scenes-redkitchen/cloud_bin_3.pth").tolist()


# 打包成 JSON
payload = {
    "src": src,
    "tgt": tgt
}

# 发送 POST 请求
print("Sending request to Predator service...")
resp = requests.post(URL, json=payload)

# 打印结果
print("Status:", resp.status_code)
print("Response JSON:")
print(json.dumps(resp.json(), indent=4))
