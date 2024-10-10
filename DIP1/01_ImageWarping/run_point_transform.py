import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 径向基函数，(||x_1-x_2||^2 + d^2) ^ alpha
def rbf(p1, p2, r):
    d = (np.linalg.norm(p1 - p2)**2 + r)**(-0.5)
    return d
# 执行仿射变换
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    image_height = image.shape[0]
    image_width = image.shape[1]

    # 构建系数矩阵
    n = source_pts.shape[0]
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A[i,j] = rbf(source_pts[i,:], source_pts[j,:], 100)
    # 解方程确定各基函数权重，满足f(src)=tar
    x = target_pts[:,0] - source_pts[:, 0]
    y = target_pts[:,1] - source_pts[:, 1]
    a = np.linalg.solve(A, x)
    b = np.linalg.solve(A, y)
    
    # 计算映射后的新坐标
    new_image = np.full((image_height, image_width, 3), 255, dtype= np.uint8)
    for i in range(image_width):
        for j in range(image_height):
            x_i = 0
            y_j = 0
            for k in range(n): 
                x_i += a[k]* rbf(np.array([i,j]), source_pts[k,:], 100)
                y_j += b[k]* rbf(np.array([i,j]), source_pts[k,:], 100) 
            x_i = int(x_i + i)
            y_j = int(y_j + j)
            if 0<= x_i < image_width and 0<= y_j< image_height:
                new_image[y_j, x_i, :] = image[j, i, :]
    # 尝试用周围像素点平均值填补空白
    for i in range(1, image_width-1):
        for j in range(1, image_height-1):
            if np.array_equal(new_image[j, i, :], np.array([255,255,255],dtype=np.uint8)):
                new_image[j, i, :] = new_image[j+1, i+1, :]/8 + new_image[j-1, i+1, :]/8 + new_image[j-1, i-1, :]/8 + new_image[j+1, i-1, :]/8 \
                    + new_image[j+1, i, :]/8 + new_image[j, i+1, :]/8 + new_image[j-1, i, :]/8 + new_image[j, i-1, :]/8

    warped_image = new_image
 #   print(image.shape)
    ### FILL: 基于MLS or RBF 实现 image warping

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
