import os
import numpy as np
import cv2
from ultralytics import YOLO, RTDETR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# ================= 配置区域 =================

MODEL_PATH = r'/home/File/wc123/RTDETR-20251008/runs/train/exp3/weights/best.pt'
IMAGES_DIR = r'/home/File/wc123/RTDETR-20251008/dataset/images/val/'
LABELS_DIR = r'/home/File/wc123/RTDETR-20251008/dataset/labels/val/'

IOU_THRESHOLD = 0.3   # 一般设置为0.5

# 原始类别只有 0-27 (28类)，但我们需要第 29 类作为背景
NUM_CLASSES = 28 
BG_INDEX = NUM_CLASSES # 28 表示背景

# ===========================================

def xywh2xyxy(x, y, w, h, img_w, img_h):
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return [x1, y1, x2, y2]

def compute_iou(box1, box2):
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
    print(f"正在加载模型: {MODEL_PATH} ...")
    try:
        model = RTDETR(MODEL_PATH)
    except:
        print("RTDETR 加载失败，尝试加载 YOLO ...")
        model = YOLO(MODEL_PATH)

    y_true_all = []
    y_pred_all = []

    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    print(f"找到 {len(image_files)} 张测试图片，开始推理...")

    for img_name in tqdm(image_files):
        img_path = os.path.join(IMAGES_DIR, img_name)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(LABELS_DIR, txt_name)

        # 1. 读取 GT
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

        # 2. 推理
        results = model.predict(img_path, verbose=False, conf=0.25)
        pred_boxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                coords = box.xyxy[0].tolist()
                pred_boxes.append({'cls': cls, 'box': coords})

        # 3. 匹配逻辑 (核心修改：处理漏检和误检)
        
        # 标记哪些预测框已经被匹配过了，防止一个预测框匹配多个GT
        pred_matched = [False] * len(pred_boxes)
        
        # 遍历 GT，寻找匹配的预测框
        for gt in gt_boxes:
            best_iou = 0
            best_pred_idx = -1
            
            # 在所有预测框中找 IoU 最大的
            for i, pred in enumerate(pred_boxes):
                # 已经被匹配过的预测框不再参与匹配（简单的一对一策略）
                if pred_matched[i]: 
                    continue
                    
                iou = compute_iou(gt['box'], pred['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = i

            if best_iou > IOU_THRESHOLD:
                # 匹配成功 (True Positive 或 错分类)
                # 记录：真实类别 -> 预测类别
                y_true_all.append(gt['cls'])
                y_pred_all.append(pred_boxes[best_pred_idx]['cls'])
                pred_matched[best_pred_idx] = True # 标记该预测框已用
            else:
                # 匹配失败 -> 漏检 (False Negative)
                # 记录：真实类别 -> 背景
                y_true_all.append(gt['cls'])
                y_pred_all.append(BG_INDEX)

        # 4. 处理剩下的预测框 -> 误检 (False Positive)
        for i, pred in enumerate(pred_boxes):
            if not pred_matched[i]:
                # 这个预测框没有匹配到任何 GT，它是凭空捏造的
                # 记录：背景 -> 预测类别
                y_true_all.append(BG_INDEX)
                y_pred_all.append(pred['cls'])

    print(f"统计完成！总样本数: {len(y_true_all)}")

    # ==========================================
    # 5. 绘制混淆矩阵
    # ==========================================
    if len(y_true_all) == 0:
        print("错误：没有收集到数据。")
        return

    # 标签列表：0~27 + "Background"
    # 总共有 29 个标签
    labels = [str(i) for i in range(NUM_CLASSES)] + ["BG"]
    
    # labels 参数需要包含 0 到 28 的所有整数
    cm = confusion_matrix(y_true_all, y_pred_all, labels=range(NUM_CLASSES + 1))

    plt.figure(figsize=(26, 22)) # 画布稍微大一点
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 9})
    
    plt.title(f'Confusion Matrix (Model Evaluation)\nTotal Samples: {len(y_true_all)}', fontsize=20)
    plt.xlabel('True Label', fontsize=15)
    plt.ylabel('Predicted Label', fontsize=15)
    
    save_path = 'dental_confusion_matrix_with_bg.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存为: {save_path}")

if __name__ == "__main__":
    main()