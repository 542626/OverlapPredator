from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import torch

# ① 加载模型（服务启动时只加载一次）
model = torch.load("kitti.pth", map_location="cpu")["state_dict"]
# 这里你可以换成真正的模型对象，例如：
# model = PredatorModel()
# model.load_state_dict(torch.load("kitti.pth")["state_dict"])
# model.eval()

class DLHandler(BaseHTTPRequestHandler):
    # ② 处理 GET 请求（测试用）
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Deep Learning Inference Server Running")

    # ③ 处理 POST 请求（推理用）
    def do_POST(self):
        # 读取请求体
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        data = json.loads(body)

        # ④ 模型推理（这里用一个假的输出示例）
        # 你可以把 data 转成 tensor，然后 model(data)
        result = {"output": "fake_result_for_demo"}

        # ⑤ 返回 JSON
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

# ⑥ 启动服务
server = HTTPServer(("0.0.0.0", 8000), DLHandler)
print("Server running on http://0.0.0.0:8000")
server.serve_forever()

