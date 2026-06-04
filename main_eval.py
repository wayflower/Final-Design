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

def create_overlay(img, binary_mask, color=(0, 255, 0), alpha=0.4):
    """生成半透明色彩覆盖效果图"""
    overlay = img.copy()
    color_layer = np.full_like(img, color, dtype=np.uint8)
    blended = cv2.addWeighted(img, 1 - alpha, color_layer, alpha, 0)
    overlay[binary_mask == 1] = blended[binary_mask == 1]
    return overlay

def batch_solve2(str_impath, str_gtpath, efg_dir="./EFG_BPP_Results", visualize_step=3, save_dir="./Output_Results"):
    """
    Python 批量测试脚本 (3行2列对称横向对比可视化)
    """
    search_pattern = os.path.join(str_impath, "*.png")
    img_files = sorted(glob.glob(search_pattern))
    im_num = len(img_files)
    
    if im_num == 0:
        print("未在路径中找到图像:", str_impath)
        return
        
    os.makedirs(save_dir, exist_ok=True)
    # [新增] 创建专门保存 3x2 对比大图的文件夹
    compare_save_dir = os.path.join(save_dir, "Comparisons")
    os.makedirs(compare_save_dir, exist_ok=True)

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
                if gtphoto is None: raise ValueError(f"GT read failed")
                gt_binary = mask2binary_weizmann2(gtphoto)
        except Exception as e:
            print(f"跳过 {img_name}: {e}")
            continue

        print(f"\ni={i+1}, 处理图像 {img_path}...")
        
        # 1. 运行自适应外扩 FSCut 算法
        start_time = time.time()
        pred_binary = segmenter.segment(imphoto)
        end_time = time.time()
        process_time = end_time - start_time
        
        f_score = 0.0
        if has_groundtruth:
            f_score, precision, recall = calc_fmeasure(gt_binary, pred_binary)
            fmeasure_table[i, 0] = f_score
            fmeasure_table[i, 1] = process_time

        # 2. 生成我方算法的半透明覆盖图
        our_overlay = create_overlay(imphoto, pred_binary, color=(0, 255, 0), alpha=0.4)
        cv2.imwrite(os.path.join(save_dir, f"{img_id}_FSCut_result.jpg"), our_overlay)

        # ==========================================================
        # [修改后] 3行2列对称排版：每张图均保存到文件夹，定时弹窗展示
        # ==========================================================
        if has_groundtruth:
            # 读取 MATLAB 导出的 EFG_BPP 结果
            efg_path = os.path.join(efg_dir, f"{img_id}_efg.png")
            if os.path.exists(efg_path):
                efg_mask = cv2.imread(efg_path, cv2.IMREAD_GRAYSCALE)
                efg_mask_bin = np.where(efg_mask > 127, 1, 0).astype(np.uint8)
                efg_score, _, _ = calc_fmeasure(gt_binary, efg_mask_bin)
                efg_mask_title = f"EFG_BPP Mask (F1: {efg_score:.3f})"
                efg_overlay = create_overlay(imphoto, efg_mask_bin, color=(0, 255, 0), alpha=0.4)
            else:
                efg_mask = np.zeros_like(pred_binary)
                efg_mask_bin = efg_mask
                efg_mask_title = "EFG_BPP Mask (Not Found)"
                efg_overlay = imphoto.copy()

            # 1. 创建并渲染 3x2 的 Matplotlib 画板
            fig = plt.figure(figsize=(14, 16))
            
            # ---------------- 第 1 行：原图 + GT ----------------
            plt.subplot(3, 2, 1)
            plt.imshow(cv2.cvtColor(imphoto, cv2.COLOR_BGR2RGB))
            plt.title("1. Original Image", fontsize=14, fontweight='bold')
            plt.axis('off')
            
            plt.subplot(3, 2, 2)
            plt.imshow(gt_binary, cmap='gray')
            plt.title("2. Ground Truth (GT)", fontsize=14, fontweight='bold')
            plt.axis('off')
            
            # ---------------- 第 2 行：EFG Mask + FSCut Mask ----------------
            plt.subplot(3, 2, 3)
            plt.imshow(efg_mask, cmap='gray')
            plt.title(f"3. {efg_mask_title}", fontsize=14, fontweight='bold')
            plt.axis('off')
            
            plt.subplot(3, 2, 4)
            plt.imshow(pred_binary, cmap='gray')
            plt.title(f"4. Our FSCut Mask (F1: {f_score:.3f})", fontsize=14, fontweight='bold')
            plt.axis('off')
            
            # ---------------- 第 3 行：EFG Overlay + FSCut Overlay ----------------
            plt.subplot(3, 2, 5)
            plt.imshow(cv2.cvtColor(efg_overlay, cv2.COLOR_BGR2RGB))
            plt.title("5. EFG_BPP Overlay", fontsize=14, fontweight='bold')
            plt.axis('off')
            
            plt.subplot(3, 2, 6)
            plt.imshow(cv2.cvtColor(our_overlay, cv2.COLOR_BGR2RGB))
            plt.title("6. Our FSCut Overlay", fontsize=14, fontweight='bold')
            plt.axis('off')
            
            plt.tight_layout()
            
            # [新增] 将整张 3x2 对比长图保存到 Output_Results/Comparisons/ 目录下
            comp_save_path = os.path.join(compare_save_dir, f"{img_id}_compare_3x2.png")
            plt.savefig(comp_save_path, bbox_inches='tight', dpi=150)
            
            # 定时弹窗展示逻辑
            if (i + 1) % visualize_step == 0:
                print(f">>> 触发 3x2 对比视图 (第 {i+1} 张) | 请手动关闭图片窗口以继续运行...")
                plt.show()
            else:
                # 如果不弹窗，必须手动关闭当前 fig，释放内存，防止后台内存泄漏内存爆炸
                plt.close(fig)

    # 统计信息打印
    if has_groundtruth:
        mean_f = np.mean(fmeasure_table[:, 0])
        std_f = np.std(fmeasure_table[:, 0])
        ci_f = std_f * 1.96 / np.sqrt(im_num)
        print("\n" + "="*40)
        print(f"Our Algorithm FinalResult: mean={mean_f:.6f}, std_err={ci_f:.6f}")
        print("="*40)

if __name__ == '__main__':
    IMAGE_DIR = "./Weizmann2Images/"
    GT_DIR = "./Weizmann2TruthOne/"
    EFG_DIR = "./EFG_BPP_Results/"  # MATLAB 导出的 EFG 结果目录
    
    # 依然是每 3 张图弹窗展示一次
    batch_solve2(IMAGE_DIR, GT_DIR, efg_dir=EFG_DIR, visualize_step=100)