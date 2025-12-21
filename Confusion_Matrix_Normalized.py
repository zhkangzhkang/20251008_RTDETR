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
IOU_THRESHOLD = 0.1 # 一般设置为0.5

# 5. 类别数量 (0-27)
NUM_CLASSES = 28

# 6. 背景类的索引 (第29类)
BG_INDEX = NUM_CLASSES 

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
    
    return interArea / float(unionArea)

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

        # --- 1. 读取 Ground Truth (真实标签) ---
        gt_boxes = [] 
        img = cv2.imread(img_path)
        if img is None: continue
        h_img, w_img = img.shape[:2]

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    cls = int(parts[0])
                    bbox = xywh2xyxy(parts[1], parts[2], parts[3], parts[4], w_img, h_img)
                    gt_boxes.append({'cls': cls, 'box': bbox})

        # --- 2. 模型推理 (Predictions) ---
        results = model.predict(img_path, verbose=False, conf=0.25)
        pred_boxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                coords = box.xyxy[0].tolist()
                pred_boxes.append({'cls': cls, 'box': coords})

        # --- 3. 匹配逻辑 (核心修改：处理 BG) ---
        
        # 记录哪些预测框已经被匹配了，防止一个预测框匹配多个GT
        matched_pred_indices = set()

        # A. 遍历 GT，寻找最佳匹配的 Pred
        for gt in gt_boxes:
            best_iou = 0
            best_pred_idx = -1
            
            # 在所有预测框中找 IoU 最大的
            for i, pred in enumerate(pred_boxes):
                if i in matched_pred_indices: continue # 跳过已被匹配的
                
                iou = compute_iou(gt['box'], pred['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = i
            
            if best_iou > IOU_THRESHOLD:
                # 匹配成功 (True Positive 或 错分类)
                y_true_all.append(gt['cls'])
                y_pred_all.append(pred_boxes[best_pred_idx]['cls'])
                matched_pred_indices.add(best_pred_idx) # 标记该预测框已使用
            else:
                # [关键] 匹配失败 = 漏检 (False Negative)
                # 真实是某类，预测为背景
                y_true_all.append(gt['cls'])
                y_pred_all.append(BG_INDEX)

        # B. 检查剩下的未匹配预测框 = 误检 (False Positive)
        for i, pred in enumerate(pred_boxes):
            if i not in matched_pred_indices:
                # 真实是背景，预测为某类
                y_true_all.append(BG_INDEX)
                y_pred_all.append(pred['cls'])

    print(f"数据统计完成！共收集到 {len(y_true_all)} 个样本(含背景交互)。")

    # ==========================================
    # 4. 绘制混淆矩阵 (归一化版，含 BG)
    # ==========================================
    if len(y_true_all) == 0:
        print("错误：没有收集到任何数据。")
        return

    # 生成标签：0~27 + "BG"
    labels = [str(i) for i in range(NUM_CLASSES)] + ["BG"]
    
    # 计算混淆矩阵，范围是 0 到 28 (共29类)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=range(NUM_CLASSES + 1))
    
    # 
    # 归一化 (按行归一化)
    # 注意：最后一行 (True Label = BG) 的总数是误检的总数。
    # 实际上 BG->BG 的数量是无法定义的(无穷大)，通常矩阵中这一格(右下角)意义不大或置0
    # 为了绘图不报错，加一个极小值
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # 某些行可能和为0（比如某类完全没出现），归一化后是0，这里保持0即可

    plt.figure(figsize=(26, 22))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 9})
    
    plt.title(f'Confusion Matrix Normalized (Model Evaluation)\nTotal Samples: {len(y_true_all)}', fontsize=20)
    plt.xlabel('True Label', fontsize=15)
    plt.ylabel('Predicted Label', fontsize=15)
    
    save_path = 'dental_confusion_matrix_bg_norm.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"含背景的归一化混淆矩阵已保存为: {save_path}")

if __name__ == "__main__":
    main()