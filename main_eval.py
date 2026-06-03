import os
import glob
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt  # [新增] 引入绘图库
from fscut_core import FSCutSegmenter

def mask2binary_weizmann2(gt_bgr):
    """
    专门针对 Weizmann2 双物体 GT 的解析函数。
    规则: 背景为灰度 (R=G=B)，前景为红色或蓝色块。
    """
    b, g, r = cv2.split(gt_bgr)
    # 找三个通道不全相等的像素，即为纯彩色前景 (红/蓝)
    is_color = np.logical_or(np.logical_or(b != g, g != r), b != r)
    binary_mask = np.zeros((gt_bgr.shape[0], gt_bgr.shape[1]), dtype=np.uint8)
    binary_mask[is_color] = 1
    return binary_mask

def calc_fmeasure(gt_binary, pred_binary):
    """
    计算 F-measure (F1-Score), Precision 和 Recall
    """
    tp = np.sum(np.logical_and(pred_binary == 1, gt_binary == 1))
    fp = np.sum(np.logical_and(pred_binary == 1, gt_binary == 0))
    fn = np.sum(np.logical_and(pred_binary == 0, gt_binary == 1))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f_score = (2 * precision * recall) / (precision + recall + 1e-8)
    return f_score, precision, recall

# [修改] 增加 visualize_step 参数，默认每 5 张图弹窗显示一次
def batch_solve2(str_impath, str_gtpath, visualize_step=10):
    """
    Python 批量测试脚本 (带定时可视化功能)
    """
    search_pattern = os.path.join(str_impath, "*.png")
    img_files = sorted(glob.glob(search_pattern))
    im_num = len(img_files)
    
    if im_num == 0:
        print("未在路径中找到图像:", str_impath)
        return
        
    print(f"找到 {im_num} 张图像准备处理。设置每 {visualize_step} 张图进行一次可视化对比...")
    
    has_groundtruth = 'weizmann2' in str_gtpath.lower()
    fmeasure_table = np.zeros((im_num, 2))
    segmenter = FSCutSegmenter() 

    for i, img_path in enumerate(img_files):
        img_name = os.path.basename(img_path)
        img_id = os.path.splitext(img_name)[0]
        
        try:
            imphoto = cv2.imread(img_path)
            if imphoto is None: raise ValueError("Image read failed")
            
            if has_groundtruth:
                gt_path = os.path.join(str_gtpath, f"{img_id}_gt.png") 
                gtphoto = cv2.imread(gt_path)
                if gtphoto is None: raise ValueError(f"GT read failed: {gt_path}")
                gt_binary = mask2binary_weizmann2(gtphoto)
        except Exception as e:
            print(f"跳过 {img_name}: {e}")
            continue

        print(f"\ni={i+1}, 处理图像 {img_path}...")
        
        start_time = time.time()
        # 核心算法调用
        pred_binary = segmenter.segment(imphoto)
        end_time = time.time()
        
        process_time = end_time - start_time
        print(f"处理完成，耗时: {process_time:.3f}s")
        
        f_score = 0.0
        if has_groundtruth:
            f_score, precision, recall = calc_fmeasure(gt_binary, pred_binary)
            fmeasure_table[i, 0] = f_score
            fmeasure_table[i, 1] = process_time
            print(f"F-measure: {f_score:.4f} (P: {precision:.4f}, R: {recall:.4f})")

        # ==========================================
        # [新增] 核心可视化逻辑 (每 n 张图触发一次)
        # ==========================================
        if has_groundtruth and (i + 1) % visualize_step == 0:
            print(f">>> 触发可视化视图 (第 {i+1} 张) | 请手动关闭图片窗口以继续运行...")
            
            plt.figure(figsize=(15, 5)) # 设置一张宽图，容纳3个子图
            
            # 1. 原始图像
            plt.subplot(1, 3, 1)
            # OpenCV 是 BGR，Matplotlib 需要 RGB，必须转换
            plt.imshow(cv2.cvtColor(imphoto, cv2.COLOR_BGR2RGB)) 
            plt.title(f"Original: {img_name}", fontsize=12)
            plt.axis('off')
            
            # 2. Ground Truth (真值)
            plt.subplot(1, 3, 2)
            plt.imshow(gt_binary, cmap='gray')
            plt.title("Ground Truth (GT)", fontsize=12)
            plt.axis('off')
            
            # 3. 算法预测结果
            plt.subplot(1, 3, 3)
            plt.imshow(pred_binary, cmap='gray')
            # 顺便把这张图的 F-score 打印在标题上，直观感受得分与视觉效果的关系
            plt.title(f"FSCut Prediction (F1: {f_score:.3f})", fontsize=12) 
            plt.axis('off')
            
            plt.tight_layout()
            
            plt.show()
            # # 非阻塞式展示，用户可以在查看完毕后手动关闭窗口继续处理下一批图像
            # plt.show(block=False)
            # plt.pause(0.1)  # 确保图像窗口能正确

    # 统计信息打印
    if has_groundtruth:
        valid_fscores = fmeasure_table[:, 0]
        valid_times = fmeasure_table[:, 1]
        
        mean_f = np.mean(valid_fscores)
        std_f = np.std(valid_fscores)
        mean_time = np.mean(valid_times)
        ci_f = std_f * 1.96 / np.sqrt(im_num)
        
        print("\n" + "="*40)
        print("最终评价报告 (FSCut - Python)")
        print("="*40)
        print(f"FinalResult   mean={mean_f:.6f}, std_err={ci_f:.6f}")
        print(f"Mean Time     ={mean_time:.6f} s/img")

if __name__ == '__main__':
    # 路径配置
    IMAGE_DIR = "./Weizmann2Images/"
    GT_DIR = "./Weizmann2TruthOne/"
    
    # 批量测试，设置 visualize_step=3，意味着每处理 3 张图弹出一个对比窗口
    batch_solve2(IMAGE_DIR, GT_DIR, visualize_step=11)