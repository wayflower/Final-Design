import os
import glob
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fscut_core import FSCutSegmenter

def mask2binary_weizmann2(gt_bgr):
    """专门针对 Weizmann2 双物体 GT 的解析函数"""
    b, g, r = cv2.split(gt_bgr)
    is_color = np.logical_or(np.logical_or(b != g, g != r), b != r)
    binary_mask = np.zeros((gt_bgr.shape[0], gt_bgr.shape[1]), dtype=np.uint8)
    binary_mask[is_color] = 1
    return binary_mask

def calc_fmeasure(gt_binary, pred_binary):
    """计算 F-measure (F1-Score), Precision 和 Recall"""
    tp = np.sum(np.logical_and(pred_binary == 1, gt_binary == 1))
    fp = np.sum(np.logical_and(pred_binary == 1, gt_binary == 0))
    fn = np.sum(np.logical_and(pred_binary == 0, gt_binary == 1))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f_score = (2 * precision * recall) / (precision + recall + 1e-8)
    return f_score, precision, recall

# [修改] 增加 save_dir 参数，指定结果保存的文件夹
def batch_solve2(str_impath, str_gtpath, visualize_step=5, save_dir="./Output_Results"):
    """
    Python 批量测试脚本 (带定时可视化与半透明结果保存功能)
    """
    search_pattern = os.path.join(str_impath, "*.png")
    img_files = sorted(glob.glob(search_pattern))
    im_num = len(img_files)
    
    if im_num == 0:
        print("未在路径中找到图像:", str_impath)
        return
        
    # [新增] 如果保存结果的文件夹不存在，则自动创建
    os.makedirs(save_dir, exist_ok=True)
    print(f"找到 {im_num} 张图像准备处理。结果将保存至: {save_dir}")
    
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
        # [新增] 半透明掩膜覆盖与结果保存逻辑
        # ==========================================
        # 1. 复制原图作为底图
        overlay_result = imphoto.copy()
        
        # 2. 创建一个纯色图层 (这里用亮绿色，BGR 格式: [0, 255, 0])
        color_layer = np.full_like(imphoto, (0, 255, 0), dtype=np.uint8)
        
        # 3. 将原图与纯色图层按比例全局混合 (原图占 60%，绿色占 40%)
        blended = cv2.addWeighted(imphoto, 0.6, color_layer, 0.4, 0)
        
        # 4. 关键：利用 numpy 掩膜替换！只在算法预测为前景的地方（pred_binary == 1），替换为混合后的半透明像素
        overlay_result[pred_binary == 1] = blended[pred_binary == 1]
        
        # 5. 保存结果图像
        # save_path = os.path.join(save_dir, f"{img_id}_FSCut_result.jpg")
        # cv2.imwrite(save_path, overlay_result)

        # ==========================================
        # 原有的阻塞式可视化逻辑 (每 n 张图触发一次)
        # ==========================================
        if has_groundtruth and (i + 1) % visualize_step == 0:
            print(f">>> 触发可视化视图 (第 {i+1} 张) | 请手动关闭图片窗口以继续运行...")
            
            # 把画板拉宽到 20，以完美容纳 4 张子图
            plt.figure(figsize=(20, 5)) 
            
            # 1. 原始图像 (Original)
            plt.subplot(1, 4, 1)
            plt.imshow(cv2.cvtColor(imphoto, cv2.COLOR_BGR2RGB)) 
            plt.title("1. Original", fontsize=12)
            plt.axis('off')
            
            # 2. 真值 (Ground Truth)
            plt.subplot(1, 4, 2)
            plt.imshow(gt_binary, cmap='gray')
            plt.title("2. Ground Truth", fontsize=12)
            plt.axis('off')
            
            # 3. 算法生成的纯色掩膜 (Prediction Mask)
            plt.subplot(1, 4, 3)
            plt.imshow(pred_binary, cmap='gray')
            # 依然在 Mask 上方显示 F1 分数，方便硬核评估
            plt.title(f"3. Pred Mask (F1: {f_score:.3f})", fontsize=12) 
            plt.axis('off')
            
            # 4. 覆盖 Mask 的半透明效果图 (Overlay Result)
            plt.subplot(1, 4, 4)
            plt.imshow(cv2.cvtColor(overlay_result, cv2.COLOR_BGR2RGB))
            plt.title("4. Overlay Result", fontsize=12) 
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()

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
        print(f"所有带透明掩膜的预测结果已保存至: {os.path.abspath(save_dir)}")

if __name__ == '__main__':
    # 路径配置
    IMAGE_DIR = "./Weizmann2Images/"
    GT_DIR = "./Weizmann2TruthOne/"
    
    # 增加了一个 save_dir 参数，默认保存在当前目录下的 Output_Results 文件夹
    batch_solve2(IMAGE_DIR, GT_DIR, visualize_step=10, save_dir="./Output_Results")