#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
点云配准结果Web可视化器
基于Three.js的浏览器可视化方案，避免X11转发卡顿问题

主要功能：
1. 从测试结果文件(.pth)中加载点云数据
2. 通过Web服务器提供数据接口
3. 利用Three.js在浏览器中可视化点云配准结果
"""

# 导入torch库，用于加载.pth格式的模型和数据文件
import torch
# 导入numpy库，用于数值计算和矩阵操作
import numpy as np
# 导入open3d库，虽然代码中没有直接使用，但通常用于点云处理
import open3d as o3d
# 导入glob库，用于文件路径的模式匹配
import glob
# 导入os库，用于路径操作和文件系统交互
import os
# 导入json库，用于序列化和反序列化JSON数据
import json
# 导入http.server库，用于创建HTTP服务器
import http.server
# 导入socketserver库，用于TCP服务器的创建和管理
import socketserver
# 导入threading库，用于多线程处理（虽然代码中没有直接使用）
import threading
# 导入webbrowser库，用于打开系统浏览器（虽然代码中没有直接使用）
import webbrowser
# 导入datetime库，用于时间戳处理（虽然代码中没有直接使用）
from datetime import datetime


# 定义点云Web可视化器类
class PointCloudWebVisualizer:
    """
    点云配准结果Web可视化器类
    
    功能：
    1. 加载点云配准的测试结果
    2. 对点云数据进行下采样处理
    3. 启动Web服务器供前端访问
    4. 生成并服务Web可视化界面
    """
    
    # 类的初始化方法
    def __init__(self, snapshot_dir, port=8000):
        """
        初始化可视化器
        
        参数：
            snapshot_dir (str): 快照目录路径，包含测试结果文件
            port (int): Web服务器的端口号，默认8000
        """
        # 存储快照目录路径
        self.snapshot_dir = snapshot_dir
        # 存储Web服务器端口号
        self.port = port
        # 初始化结果列表，用于存储加载的所有样本
        self.results = []
        # 初始化当前样本索引
        self.current_sample = 0
    
    # 加载测试结果的方法
    def load_test_results(self, max_samples=20, downsample_factor=10):
        """
        加载测试结果文件
        
        功能：
        1. 查找benchmark路径中的所有.pth文件
        2. 加载每个文件中的点云数据
        3. 提取源点云和目标点云
        4. 应用旋转和平移变换到源点云
        5. 对点云进行下采样以减少数据量
        
        参数：
            max_samples (int): 最多加载多少个样本，默认20
            downsample_factor (int): 下采样因子，每downsample_factor个点取一个，默认10
        
        返回：
            bool: 是否成功加载至少一个样本
        """
        # 构建benchmark路径
        benchmark_path = os.path.join(self.snapshot_dir, "indoor", "3DMatch")
        
        # 检查路径是否存在
        if not os.path.exists(benchmark_path):
            # 如果路径不存在，打印错误信息
            print("❌ 测试结果目录不存在:", benchmark_path)
            return False
        
        # 查找路径中所有.pth文件
        result_files = glob.glob(os.path.join(benchmark_path, "*.pth"))
        # 打印找到的文件数量
        print(f"✅ 找到 {len(result_files)} 个测试结果文件")
        
        # 如果没有找到任何.pth文件，返回False
        if len(result_files) == 0:
            print("❌ 未找到.pth文件")
            return False
        
        # 循环加载指定数量的样本
        for i, result_file in enumerate(result_files[:max_samples]):
            try:
                # 打印当前加载的样本信息
                print(f"📥 加载样本 {i+1}: {os.path.basename(result_file)}")
                # 使用torch加载.pth文件
                data = torch.load(result_file)
                
                # ===== 数据提取部分 =====
                # 从data字典中提取点云数据（numpy数组格式）
                pcd = data['pcd'].numpy()
                # 提取源点云的长度（区分源和目标点云）
                len_src = data['len_src']
                # 提取旋转矩阵（numpy数组格式）
                rot = data['rot'].numpy()
                # 提取平移向量（numpy数组格式）
                trans = data['trans'].numpy()
                
                # ===== 点云分离部分 =====
                # 根据len_src分离源点云（索引0到len_src）
                src_pcd = pcd[:len_src]
                # 分离目标点云（从len_src开始到末尾）
                tgt_pcd = pcd[len_src:]
                
                # ===== 下采样部分 =====
                # 对源点云进行下采样，步长为downsample_factor
                src_points = src_pcd[::downsample_factor]
                # 对目标点云进行下采样，步长为downsample_factor
                tgt_points = tgt_pcd[::downsample_factor]
                
                # ===== 构建变换矩阵部分 =====
                # 创建4x4单位矩阵作为变换矩阵的基础
                transform_matrix = np.eye(4)
                # 将旋转矩阵填入变换矩阵的左上角3x3部分
                transform_matrix[:3, :3] = rot
                # 将平移向量填入变换矩阵的右上角3x1部分
                transform_matrix[:3, 3] = trans.flatten()
                
                # ===== 应用变换部分 =====
                # 将旋转矩阵应用到源点云，然后加上平移向量得到变换后的点云
                # 点云的形状为(N, 3)，需要转置为(3, N)进行矩阵乘法，再转置回(N, 3)
                src_transformed = (rot @ src_points.T + trans).T
                
                # ===== 构建样本数据部分 =====
                # 创建字典存储当前样本的所有信息
                sample_data = {
                    # 样本ID
                    'sample_id': i,
                    # 原始文件名
                    'filename': os.path.basename(result_file),
                    # 源点云坐标列表（转换为Python列表格式以便JSON序列化）
                    'source_points': src_points.tolist(),
                    # 目标点云坐标列表
                    'target_points': tgt_points.tolist(),
                    # 变换后的源点云坐标列表
                    'source_transformed': src_transformed.tolist(),
                    # 变换信息字典
                    'transform': {
                        # 旋转矩阵（3x3）
                        'rotation': rot.tolist(),
                        # 平移向量（3x1）
                        'translation': trans.tolist(),
                        # 完整的变换矩阵（4x4）
                        'matrix': transform_matrix.tolist()
                    },
                    # 统计信息字典
                    'stats': {
                        # 原始源点云的点数
                        'source_original': len(src_pcd),
                        # 原始目标点云的点数
                        'target_original': len(tgt_pcd),
                        # 下采样后源点云的点数
                        'source_downsampled': len(src_points),
                        # 下采样后目标点云的点数
                        'target_downsampled': len(tgt_points),
                        # 使用的下采样因子
                        'downsample_factor': downsample_factor
                    }
                }
                
                # 将样本数据添加到results列表
                self.results.append(sample_data)
                # 打印加载成功信息，显示点云点数
                print(f"   ✅ 加载成功: {len(src_points)}源点, {len(tgt_points)}目标点")
                
            # 异常处理：捕获加载过程中的所有异常
            except Exception as e:
                # 打印错误信息并继续加载下一个样本
                print(f"   ❌ 加载失败: {e}")
                continue
            
        # 打印成功加载的样本总数    
        print(f"✅ 成功加载 {len(self.results)} 个样本")
        # 返回是否至少成功加载了一个样本
        return len(self.results) > 0
    
    # 生成Web界面的方法
    def generate_web_interface(self):
        """
        生成完整的Web界面
        
        功能：
        1. 尝试读取优化后的HTML文件
        2. 如果不存在则使用默认HTML内容
        3. 将HTML文件写入当前目录
        """
        
        # ===== 读取HTML文件部分 =====
        # 构建HTML文件的路径（与当前Python文件在同一目录）
        html_file_path = os.path.join(os.path.dirname(__file__), "web_visualizer.html")
        
        # 检查HTML文件是否存在
        if os.path.exists(html_file_path):
            # 打开并读取HTML文件
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            # 打印成功读取的信息
            print("✅ 使用优化后的HTML界面文件")
        else:
            # 如果文件不存在，使用硬编码的默认HTML内容
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
            # 打印使用默认内容的信息
            print("⚠️ 使用默认HTML内容")
        
        # ===== 写入HTML文件部分 =====
        # 以写入模式打开输出文件，UTF-8编码
        with open("pointcloud_visualizer.html", "w", encoding="utf-8") as f:
            # 将HTML内容写入文件
            f.write(html_content)
        
        # 打印生成成功的信息
        print("✅ Web界面已生成: pointcloud_visualizer.html")
    
    # 启动Web服务器的方法
    def start_web_server(self):
        """
        启动Web服务器
        
        功能：
        1. 创建自定义HTTP请求处理程序
        2. 处理'/data'路由返回JSON格式的点云数据
        3. 处理其他路由返回静态文件
        4. 启动TCP服务器监听指定端口
        """
        
        # ===== 定义自定义HTTP请求处理器类 =====
        class WebVisualizerHandler(http.server.SimpleHTTPRequestHandler):
            """
            自定义HTTP请求处理器
            继承自SimpleHTTPRequestHandler，扩展其功能
            """
            
            # 处理HTTP GET请求的方法
            def do_GET(self):
                """
                处理GET请求
                
                功能：
                1. 如果请求路径为'/data'，返回JSON格式的点云数据
                2. 否则调用父类方法返回静态文件
                """
                # 检查请求路径是否为'/data'
                if self.path == '/data':
                    # 发送HTTP响应状态码200（成功）
                    self.send_response(200)
                    # 设置响应头：内容类型为JSON
                    self.send_header('Content-type', 'application/json')
                    # 设置CORS头，允许跨域请求
                    self.send_header('Access-Control-Allow-Origin', '*')
                    # 结束响应头部分
                    self.end_headers()
                    # 将self.results列表转换为JSON字符串，编码为字节后写入响应体
                    self.wfile.write(json.dumps(self.results).encode())
                else:
                    # 如果不是'/data'路由，调用父类方法处理静态文件请求
                    super().do_GET()
        
        # ===== 配置处理器部分 =====
        # 将类变量self.results赋给处理器类，使处理器可以访问点云数据
        WebVisualizerHandler.results = self.results
        
        # ===== 改变工作目录部分 =====
        # 改变工作目录到项目根目录，以便正确服务HTML文件
        os.chdir('/media/user/新加卷/ljn_worksp/OverlapPredator-main')
        
        # ===== 创建并启动TCP服务器部分 =====
        # 创建TCPServer实例，绑定到localhost和指定端口
        # 使用with语句确保服务器正确关闭
        with socketserver.TCPServer(("localhost", self.port), WebVisualizerHandler) as httpd:
            # 打印服务器启动成功信息
            print(f"🌐 Web服务器启动成功!")
            # 打印访问地址
            print(f"📍 访问地址: http://localhost:{self.port}/pointcloud_visualizer.html")
            # 打印已加载的点云样本数量
            print(f"📊 已加载 {len(self.results)} 个点云样本")
            # 打印停止服务器的提示
            print("⏹️  按 Ctrl+C 停止服务器")
            # 打印分隔线
            print("-" * 50)
            
            # ===== 服务器主循环部分 =====
            try:
                # 启动服务器的阻塞式主循环，处理请求直到被中断
                httpd.serve_forever()
            # 捕获键盘中断异常（Ctrl+C）
            except KeyboardInterrupt:
                # 打印服务器停止信息
                print("\n🛑 服务器已停止")


# ===== 主函数定义 =====
def main():
    """
    主函数
    
    功能：
    1. 创建可视化器实例
    2. 加载测试结果
    3. 生成Web界面
    4. 启动Web服务器
    """
    
    # ===== 初始化部分 =====
    # 设置快照目录路径，包含测试结果文件
    snapshot_dir = "/media/user/新加卷/ljn_worksp/OverlapPredator-main/snapshot"
    
    # 创建PointCloudWebVisualizer类的实例
    visualizer = PointCloudWebVisualizer(snapshot_dir)
    
    # ===== 执行可视化流程部分 =====
    # 尝试加载测试结果
    if visualizer.load_test_results():
        # 如果成功加载结果，生成Web界面
        visualizer.generate_web_interface()
        
        # 启动Web服务器
        visualizer.start_web_server()
    else:
        # 如果加载失败，打印错误信息
        print("❌ 无法加载测试结果，请检查快照目录")


# ===== 程序入口 =====
# 检查当前脚本是否作为主程序运行（而不是被导入）
if __name__ == "__main__":
    # 调用主函数
    main()
