import numpy as np
import cv2
from skimage.segmentation import felzenszwalb
from skimage import graph
from skimage.color import rgb2gray
import maxflow

class FSCutSegmenter:
    def __init__(self, fz_scale=100, fz_sigma=0.5, fz_min_size=50):
        """
        Felzenszwalb-Saliency Cut 分割器
        """
        self.fz_scale = fz_scale
        self.fz_sigma = fz_sigma
        self.fz_min_size = fz_min_size

    def spectral_residual_saliency(self, image):
        """
        频域残差显著性先验 (Spectral Residual)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_float = np.float32(gray) / 255.0
        
        # 傅里叶变换
        dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
        mag, phase = cv2.cartToPolar(dft[:, :, 0], dft[:, :, 1])
        
        # 提取对数振幅谱并使用均值滤波求残差
        log_mag = np.log(mag + 1e-9)
        blur_mag = cv2.blur(log_mag, (3, 3))
        spectral_residual = log_mag - blur_mag
        
        # 反变换回空域
        complex_res = np.zeros_like(dft)
        complex_res[:, :, 0], complex_res[:, :, 1] = cv2.polarToCart(np.exp(spectral_residual), phase)
        idft = cv2.idft(complex_res)
        saliency = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
        
        # 高斯平滑并归一化
        saliency = cv2.GaussianBlur(saliency, (9, 9), 2.5)
        saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)
        return saliency

    def segment(self, image):
        """
        执行完整的前景分割 (加入边界先验与双目标拓扑约束)
        """
        # 1. 频域显著性先验
        saliency_map = self.spectral_residual_saliency(image)
        
        # 2. Felzenszwalb 超像素提取
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        segments = felzenszwalb(rgb_img, scale=self.fz_scale, sigma=self.fz_sigma, min_size=self.fz_min_size)
        num_nodes = np.max(segments) + 1
        
        # ==========================================
        # [新增约束 1] 提取图像边界上的超像素 (边界背景先验)
        # ==========================================
        boundary_mask = np.zeros_like(segments, dtype=bool)
        boundary_mask[0, :] = True      # 顶边
        boundary_mask[-1, :] = True     # 底边
        boundary_mask[:, 0] = True      # 左边
        boundary_mask[:, -1] = True     # 右边
        # 找到所有与边界有交集的超像素 ID
        boundary_segments = np.unique(segments[boundary_mask])
        
        # 3. 计算超像素级别的数据项，并引入 Otsu 自适应阈值
        node_saliency = np.zeros(num_nodes)
        for i in range(num_nodes):
            mask = (segments == i)
            node_saliency[i] = np.mean(saliency_map[mask])
            
        sal_uint8 = (saliency_map * 255).astype(np.uint8)
        otsu_thresh_val, _ = cv2.threshold(sal_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_thresh = otsu_thresh_val / 255.0
        
        # 4. 构建图割 (Graph Cut)
        rag = graph.rag_mean_color(rgb_img, segments)
        g = maxflow.Graph[float]()
        nodes = g.add_nodes(num_nodes)
        
        # 添加数据项边
        for i in range(num_nodes):
            sal_val = node_saliency[i]
            
            if i in boundary_segments:
                # [核心逻辑] 如果挨着图片边缘，强制判定为背景！彻底消灭背景反转。
                weight_fg, weight_bg = 0.0, 1000.0
            elif sal_val > otsu_thresh * 1.1:
                weight_fg, weight_bg = 1000.0, 0.0
            elif sal_val < otsu_thresh * 0.8:
                weight_fg, weight_bg = 0.0, 1000.0
            else:
                weight_fg = sal_val * 10.0
                weight_bg = (1.0 - sal_val) * 10.0
                
            g.add_tedge(nodes[i], weight_fg, weight_bg)
            
        # 添加平滑项边
        gamma = 900.0 
        lambda_smooth = 1.0 
        for edge in rag.edges:
            n1, n2 = edge
            color_diff = rag[n1][n2]['weight'] 
            smooth_weight = lambda_smooth * np.exp(- (color_diff ** 2) / gamma)
            g.add_edge(nodes[n1], nodes[n2], smooth_weight, smooth_weight)
            
        # 5. 求解最大流最小割并生成初步掩膜
        g.maxflow()
        binary_mask = np.zeros_like(segments, dtype=np.uint8)
        for i in range(num_nodes):
            if g.get_segment(nodes[i]) == 0: 
                binary_mask[segments == i] = 1
                
        # ==========================================
        # [新增约束 2] 利用你的关键信息：双目标拓扑约束
        # ==========================================
        # 寻找所有独立连通的白色前景块
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        final_mask = np.zeros_like(binary_mask)
        
        # num_labels 包含了背景(label 0)，所以如果 >= 3，说明找到了至少 2 个前景块
        if num_labels >= 3:
            # 提取所有前景块的面积 (跳过 stats[0]，因为那是背景的面积)
            areas = stats[1:, cv2.CC_STAT_AREA]
            # 找到面积最大的前 2 个连通域的索引
            # 注意：argsort 返回的是从小到大的索引，所以取最后两个 [-2:]
            # 索引 + 1 是为了对应回原来的 label 编号
            top2_idx = np.argsort(areas)[-2:] + 1
            
            # 强制只保留这 2 个最大面积的前景，其余噪点全部抹除
            final_mask[labels == top2_idx[0]] = 1
            final_mask[labels == top2_idx[1]] = 1
        else:
            # 极端情况兜底：如果连 2 个目标都没找够，就保留原样
            final_mask = binary_mask 
                
        return final_mask