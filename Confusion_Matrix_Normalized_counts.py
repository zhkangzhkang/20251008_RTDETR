import os
import numpy as np
import cv2
from ultralytics import YOLO, RTDETR 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# ================= 配置区域 (请修改这里) =================

# 1. 你的模型路径
MODEL_PATH = r'/home/File/wc123/RTDETR-20251008/runs/train/exp3/weights/best.pt' 

# 2. 测试集图片文件夹
IMAGES_DIR = r'/home/File/wc123/RTDETR-20251008/dataset/images/val/'

# 3. 对应文件名的标签文件夹
LABELS_DIR = r'/home/File/wc123/RTDETR-20251008/dataset/labels/val/'

# 4. IoU 阈值
IOU_THRESHOLD = 0.7 

# 5. 类别数量
NUM_CLASSES = 28

# =======================================================

def xywh2xyxy(x, y, w, h, img_w, img_h):
    """将归一化中心坐标转为绝对角点坐标"""
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return [x1, y1, x2, y2]

def compute_iou(box1, box2):
    """计算两个框的 IoU"""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    unionArea = box1Area + box2Area - interArea
    if unionArea == 0: return 0
    
    iou = interArea / float(unionArea)
    return iou

def main():
    # 1. 加载模型
    print(f"正在加载模型: {MODEL_PATH} ...")
    try:
        model = RTDETR(MODEL_PATH)
    except:
        print("RTDETR 加载失败或未导入，尝试加载 YOLO 模型...")
        model = YOLO(MODEL_PATH)

    y_true_all = []
    y_pred_all = []

    # 获取所有图片列表
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    print(f"找到 {len(image_files)} 张测试图片，开始推理...")

    for img_name in tqdm(image_files):
        img_path = os.path.join(IMAGES_DIR, img_name)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(LABELS_DIR, txt_name)

        # 1. 读取 Ground Truth
        gt_boxes = [] 
        if os.path.exists(txt_path):
            img = cv2.imread(img_path)
            if img is None: continue
            h_img, w_img = img.shape[:2]

            with open(txt_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    cls = int(parts[0])
                    bbox = xywh2xyxy(parts[1], parts[2], parts[3], parts[4], w_img, h_img)
                    gt_boxes.append({'cls': cls, 'box': bbox})
        else:
            continue

        # 2. 模型推理
        results = model.predict(img_path, verbose=False, conf=0.25)
        pred_boxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                coords = box.xyxy[0].tolist()
                pred_boxes.append({'cls': cls, 'box': coords})

        # 3. 匹配
        for gt in gt_boxes:
            best_iou = 0
            best_match_cls = -1
            
            for pred in pred_boxes:
                iou = compute_iou(gt['box'], pred['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_match_cls = pred['cls']
            
            if best_iou > IOU_THRESHOLD:
                y_true_all.append(gt['cls'])      
                y_pred_all.append(best_match_cls) 
            else:
                pass 

    print(f"数据统计完成！共收集到 {len(y_true_all)} 个有效匹配样本。")

    # ==========================================
    # 4. 绘制混淆矩阵 (数量版)
    # ==========================================
    if len(y_true_all) == 0:
        print("错误：没有收集到任何数据。")
        return

    labels = [str(i) for i in range(NUM_CLASSES)]
    
    # 计算混淆矩阵 (不进行归一化)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=range(NUM_CLASSES))

    plt.figure(figsize=(24, 20))
    
    # 关键修改点：
    # 1. data=cm (直接传原始计数矩阵)
    # 2. fmt='d' (格式化为整数 digit)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 9})
    
    plt.title(f'Confusion Matrix Counts (Model Evaluation)\nTotal Samples: {len(y_true_all)}', fontsize=20)
    plt.ylabel('True Label', fontsize=15)
    plt.xlabel('Predicted Label', fontsize=15)
    
    save_path = 'Confusion_Matrix_Normalized_counts.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵(数量版)已保存为: {save_path}")

if __name__ == "__main__":
    main()