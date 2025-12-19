#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
点云配准结果Web可视化器
基于Three.js的浏览器可视化方案，避免X11转发卡顿问题
"""

import torch
import numpy as np
import open3d as o3d
import glob
import os
import json
import http.server
import socketserver
import threading
import webbrowser
from datetime import datetime


class PointCloudWebVisualizer:
    def __init__(self, snapshot_dir, port=8000):
        self.snapshot_dir = snapshot_dir
        self.port = port
        self.results = []
        self.current_sample = 0
        
    def load_test_results(self, max_samples=20, downsample_factor=10):
        """加载测试结果文件"""
        benchmark_path = os.path.join(self.snapshot_dir, "indoor", "3DMatch")
        
        if not os.path.exists(benchmark_path):
            print("❌ 测试结果目录不存在:", benchmark_path)
            return False
            
        result_files = glob.glob(os.path.join(benchmark_path, "*.pth"))
        print(f"✅ 找到 {len(result_files)} 个测试结果文件")
        
        if len(result_files) == 0:
            print("❌ 未找到.pth文件")
            return False
        
        # 加载指定数量的样本
        for i, result_file in enumerate(result_files[:max_samples]):
            try:
                print(f"📥 加载样本 {i+1}: {os.path.basename(result_file)}")
                data = torch.load(result_file)
                
                # 提取数据
                pcd = data['pcd'].numpy()
                len_src = data['len_src']
                rot = data['rot'].numpy()
                trans = data['trans'].numpy()
                
                # 分离点云
                src_pcd = pcd[:len_src]
                tgt_pcd = pcd[len_src:]
                
                # 下采样以减少数据量
                src_points = src_pcd[::downsample_factor]
                tgt_points = tgt_pcd[::downsample_factor]
                
                # 构建变换矩阵
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rot
                transform_matrix[:3, 3] = trans.flatten()
                
                # 应用变换到源点云
                src_transformed = (rot @ src_points.T + trans).T
                
                sample_data = {
                    'sample_id': i,
                    'filename': os.path.basename(result_file),
                    'source_points': src_points.tolist(),
                    'target_points': tgt_points.tolist(),
                    'source_transformed': src_transformed.tolist(),
                    'transform': {
                        'rotation': rot.tolist(),
                        'translation': trans.tolist(),
                        'matrix': transform_matrix.tolist()
                    },
                    'stats': {
                        'source_original': len(src_pcd),
                        'target_original': len(tgt_pcd),
                        'source_downsampled': len(src_points),
                        'target_downsampled': len(tgt_points),
                        'downsample_factor': downsample_factor
                    }
                }
                
                self.results.append(sample_data)
                print(f"   ✅ 加载成功: {len(src_points)}源点, {len(tgt_points)}目标点")
                
            except Exception as e:
                print(f"   ❌ 加载失败: {e}")
                continue
                
        print(f"✅ 成功加载 {len(self.results)} 个样本")
        return len(self.results) > 0
    
    def generate_web_interface(self):
        """生成完整的Web界面"""
        
        # 读取优化后的HTML文件
        html_file_path = os.path.join(os.path.dirname(__file__), "web_visualizer.html")
        
        if os.path.exists(html_file_path):
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            print("✅ 使用优化后的HTML界面文件")
        else:
            # 如果文件不存在，使用默认的HTML内容
            html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>点云配准结果可视化</title>
    <meta charset="utf-8">
    <style>
        body { 
            margin: 0; 
            padding: 0; 
            font-family: Arial, sans-serif; 
            background: #1e1e1e; 
            color: white; 
            overflow: hidden;
            height: 100vh;
        }
        .container { 
            display: flex; 
            flex-direction: column; 
            height: 100vh; 
        }
        .controls { 
            padding: 8px 12px; 
            background: #2d2d2d; 
            border-bottom: 1px solid #444;
            min-height: 60px;
            flex-shrink: 0;
        }
        .viewer { 
            flex: 1; 
            background: #000; 
            position: relative;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .title {
            font-size: 14px;
            font-weight: bold;
            color: #007acc;
        }
        .button-group {
            display: flex;
            gap: 6px;
            align-items: center;
        }
        button { 
            padding: 4px 8px; 
            background: #007acc; 
            color: white; 
            border: none; 
            border-radius: 2px; 
            cursor: pointer; 
            font-size: 12px;
            min-width: 60px;
        }
        button:hover { background: #005a9e; }
        .info { 
            display: flex;
            gap: 15px;
            font-size: 11px;
            color: #ccc;
        }
        .sample-info {
            font-size: 11px;
            color: #aaa;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="controls">
            <div class="header">
                <div class="title">点云配准结果可视化</div>
                <div class="button-group">
                    <button onclick="prevSample()">上一个</button>
                    <button onclick="nextSample()">下一个</button>
                    <span class="sample-info" id="sampleInfo">样本 0/0</span>
                </div>
            </div>
            <div class="info">
                <div>蓝色: 目标点云</div>
                <div>黄色: 变换后的源点云</div>
                <div id="pointInfo">点数: 源:0, 目标:0</div>
            </div>
        </div>
        <div id="viewer" class="viewer"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        let currentSample = 0;
        let samples = [];
        let scene, camera, renderer, controls;
        
        // 初始化Three.js场景
        function initThreeJS() {
            const container = document.getElementById('viewer');
            
            // 创建场景
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1e1e1e);
            
            // 创建相机
            camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.z = 5;
            
            // 创建渲染器
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(renderer.domElement);
            
            // 添加轨道控制
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.25;
            
            // 添加光源
            const ambientLight = new THREE.AmbientLight(0x404040);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);
            
            // 添加坐标轴
            const axesHelper = new THREE.AxesHelper(2);
            scene.add(axesHelper);
            
            // 响应窗口大小变化
            window.addEventListener('resize', onWindowResize);
            
            animate();
        }
        
        function onWindowResize() {
            const container = document.getElementById('viewer');
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }
        
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        
        // 加载样本数据
        function loadSamples() {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    samples = data;
                    if (samples.length > 0) {
                        currentSample = 0;
                        displaySample(currentSample);
                    }
                })
                .catch(error => console.error('加载数据失败:', error));
        }
        
        // 显示样本
        function displaySample(index) {
            // 清空场景
            while(scene.children.length > 0){ 
                scene.remove(scene.children[0]); 
            }
            
            const sample = samples[index];
            
            // 更新界面信息
            document.getElementById('sampleInfo').textContent = 
                `样本 ${index + 1}/${samples.length}`;
            document.getElementById('pointInfo').textContent = 
                `点数: 源: ${sample.source_points.length}, 目标: ${sample.target_points.length}`;
            
            // 创建目标点云（蓝色）
            const targetGeometry = new THREE.BufferGeometry();
            const targetVertices = new Float32Array(sample.target_points.flat());
            targetGeometry.setAttribute('position', new THREE.BufferAttribute(targetVertices, 3));
            const targetMaterial = new THREE.PointsMaterial({ 
                color: 0x007acc, 
                size: 0.02,
                sizeAttenuation: true
            });
            const targetPoints = new THREE.Points(targetGeometry, targetMaterial);
            scene.add(targetPoints);
            
            // 创建源点云（黄色）
            const sourceGeometry = new THREE.BufferGeometry();
            const sourceVertices = new Float32Array(sample.source_points.flat());
            sourceGeometry.setAttribute('position', new THREE.BufferAttribute(sourceVertices, 3));
            const sourceMaterial = new THREE.PointsMaterial({ 
                color: 0xffcc00, 
                size: 0.02,
                sizeAttenuation: true
            });
            const sourcePoints = new THREE.Points(sourceGeometry, sourceMaterial);
            scene.add(sourcePoints);
            
            // 添加坐标轴
            const axesHelper = new THREE.AxesHelper(2);
            scene.add(axesHelper);
            
            // 添加光源
            const ambientLight = new THREE.AmbientLight(0x404040);
            scene.add(ambientLight);
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);
        }
        
        function nextSample() {
            if (currentSample < samples.length - 1) {
                currentSample++;
                displaySample(currentSample);
            }
        }
        
        function prevSample() {
            if (currentSample > 0) {
                currentSample--;
                displaySample(currentSample);
            }
        }
        
        // 页面加载完成后初始化
        window.onload = function() {
            initThreeJS();
            loadSamples();
        };
    </script>
</body>
</html>'''
            print("⚠️ 使用默认HTML内容")
        
        # 写入HTML文件
        with open("pointcloud_visualizer.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print("✅ Web界面已生成: pointcloud_visualizer.html")
    
    def start_web_server(self):
        """启动Web服务器"""
        
        class WebVisualizerHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/data':
                    # 返回JSON数据
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(self.results).encode())
                else:
                    # 服务静态文件
                    super().do_GET()
        
        # 设置自定义处理程序
        WebVisualizerHandler.results = self.results
        
        os.chdir('/media/user/新加卷/ljn_worksp/OverlapPredator-main')
        
        # 使用localhost绑定，避免外部访问问题
        with socketserver.TCPServer(("localhost", self.port), WebVisualizerHandler) as httpd:
            print(f"🌐 Web服务器启动成功!")
            print(f"📍 访问地址: http://localhost:{self.port}/pointcloud_visualizer.html")
            print(f"📊 已加载 {len(self.results)} 个点云样本")
            print("⏹️  按 Ctrl+C 停止服务器")
            print("-" * 50)
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n🛑 服务器已停止")


def main():
    """主函数"""
    # 设置快照目录路径
    snapshot_dir = "/media/user/新加卷/ljn_worksp/OverlapPredator-main/snapshot"
    
    # 创建可视化器实例
    visualizer = PointCloudWebVisualizer(snapshot_dir)
    
    # 加载测试结果
    if visualizer.load_test_results():
        # 生成Web界面
        visualizer.generate_web_interface()
        
        # 启动Web服务器
        visualizer.start_web_server()
    else:
        print("❌ 无法加载测试结果，请检查快照目录")


if __name__ == "__main__":
    main()