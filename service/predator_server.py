from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from easydict import EasyDict as edict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(ROOT_DIR)

# 项目内部模块
from models.architectures import KPFCNN
from datasets.indoor import IndoorDataset
from datasets.dataloader import get_dataloader
from lib.utils import load_obj, load_config
from lib.benchmark_utils import ransac_pose_estimation

###############################################
# 1. 加载 config + 模型 + neighborhood_limits
###############################################
print("Loading Predator config and model...")


config_path = os.path.join(ROOT_DIR, "configs/test/indoor.yaml")

config = load_config(config_path)
config = edict(config)

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构造模型结构
config.architecture = ["simple", "resnetb"]
for i in range(config.num_layers - 1):
    config.architecture.append("resnetb_strided")
    config.architecture.append("resnetb")
    config.architecture.append("resnetb")
for i in range(config.num_layers - 2):
    config.architecture.append("nearest_upsample")
    config.architecture.append("unary")
config.architecture.append("nearest_upsample")
config.architecture.append("last_unary")

# 初始化模型
config.model = KPFCNN(config).to(config.device)

# 加载预训练权重
state = torch.load(config.pretrain, map_location=config.device)
config.model.load_state_dict(state["state_dict"])
config.model.eval()

# 计算 neighborhood_limits
info_train = load_obj(config.train_info)
train_set = IndoorDataset(info_train, config, data_augmentation=False)
_, neighborhood_limits = get_dataloader(
    dataset=train_set,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=1,
)
config.neighborhood_limits = neighborhood_limits

print("Predator model ready.")

###############################################
# 2. ThreeDMatchDemo
###############################################
class ThreeDMatchDemo(Dataset):
    def __init__(self, config, src_points, tgt_points):
        super(ThreeDMatchDemo, self).__init__()
        self.config = config
        self.src_pcd = np.asarray(src_points, dtype=np.float32)
        self.tgt_pcd = np.asarray(tgt_points, dtype=np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, item):
        src_pcd = self.src_pcd
        tgt_pcd = self.tgt_pcd

        src_feats = np.ones_like(src_pcd[:, :1], dtype=np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1], dtype=np.float32)

        rot = np.eye(3, dtype=np.float32)
        trans = np.ones((3, 1), dtype=np.float32)
        correspondences = torch.ones(1, 2).long()

        return (
            src_pcd,
            tgt_pcd,
            src_feats,
            tgt_feats,
            rot,
            trans,
            correspondences,
            src_pcd,
            tgt_pcd,
            torch.ones(1, dtype=torch.float32),
        )

###############################################
# 3. predator_infer
###############################################
def predator_infer(config, src_points, tgt_points):
    """
    输入：
        src_points, tgt_points: [[x,y,z], ...]
    输出：
        dict {
            "transform": 4x4 刚体变换矩阵（list[list[float]]） 或 None
        }
    """
    # 1. 构造只含一对点云的 DemoDataset 和 DataLoader
    demo_set = ThreeDMatchDemo(config, src_points, tgt_points)
    demo_loader, _ = get_dataloader(
        dataset=demo_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        neighborhood_limits=config.neighborhood_limits,
    )

    c_loader_iter = iter(demo_loader)
    inputs = next(c_loader_iter)

    # 2. 把 batch 移到 device（完全照官方 main）
    for k, v in inputs.items():
        if isinstance(v, list):
            inputs[k] = [item.to(config.device) for item in v]
        else:
            inputs[k] = v.to(config.device)

    # 3. 前向推理
    with torch.no_grad():
        feats, scores_overlap, scores_saliency = config.model(inputs)

    pcd = inputs["points"][0]
    len_src = inputs["stack_lengths"][0][0]

    src_pcd = pcd[:len_src]
    tgt_pcd = pcd[len_src:]

    src_raw = src_pcd.clone()
    tgt_raw = tgt_pcd.clone()

    src_feats = feats[:len_src].detach().cpu()
    tgt_feats = feats[len_src:].detach().cpu()
    src_overlap = scores_overlap[:len_src].detach().cpu()
    tgt_overlap = scores_overlap[len_src:].detach().cpu()
    src_saliency = scores_saliency[:len_src].detach().cpu()
    tgt_saliency = scores_saliency[len_src:].detach().cpu()

    # 4. 按 overlap * saliency 做概率采样（完全照官方 main）
    src_scores = src_overlap * src_saliency
    tgt_scores = tgt_overlap * tgt_saliency

    if src_pcd.size(0) > config.n_points:
        idx = np.arange(src_pcd.size(0))
        probs = (src_scores / src_scores.sum()).numpy().flatten()
        idx = np.random.choice(idx, size=config.n_points, replace=False, p=probs)
        src_pcd = src_pcd[idx]
        src_feats = src_feats[idx]

    if tgt_pcd.size(0) > config.n_points:
        idx = np.arange(tgt_pcd.size(0))
        probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
        idx = np.random.choice(idx, size=config.n_points, replace=False, p=probs)
        tgt_pcd = tgt_pcd[idx]
        tgt_feats = tgt_feats[idx]

    # 5. RANSAC 估计位姿（完全照官方 main）
    tsfm = ransac_pose_estimation(
        src_pcd,
        tgt_pcd,
        src_feats,
        tgt_feats,
        mutual=False,
    )

    # 6. 处理失败情况 + 统一转 numpy
    if tsfm is None:
        print(">>> RANSAC failed, return None")
        return {"transform": None}

    if isinstance(tsfm, torch.Tensor):
        tsfm = tsfm.cpu().numpy()
    else:
        tsfm = np.asarray(tsfm)

    return {
        "transform": tsfm.tolist()
    }

###############################################
# 4. HTTP Handler
###############################################
class PredatorHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        msg = "Predator point cloud registration service".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain;charset=utf-8")
        self.send_header("Content-Length", str(len(msg)))
        self.end_headers()
        self.wfile.write(msg)

    def do_POST(self):
        print(">>> Received POST")
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        data = json.loads(body.decode("utf-8"))

        src_points = data["src"]
        tgt_points = data["tgt"]

        result = predator_infer(config, src_points, tgt_points)
        print(">>> result from predator_infer:", result)

        resp = json.dumps(result).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json;charset=utf-8")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

###############################################
# 5. 启动服务
###############################################
server = HTTPServer(("0.0.0.0", 8000), PredatorHandler)
print("Predator service running on port 8000...")
server.serve_forever()
