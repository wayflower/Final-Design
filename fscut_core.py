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
        H, W = image.shape[:2]
        
        # ==========================================
        # 1. 频域显著性 + 暴力物理膨胀 (撑大基础盘)
        # ==========================================
        saliency_map = self.spectral_residual_saliency(image)
        
        # [核心修改 1] 用一个巨大的核，把显著性高亮区强行向外推延！
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        saliency_map = cv2.dilate(saliency_map, dilate_kernel)
        saliency_map = cv2.GaussianBlur(saliency_map, (15, 15), 0) # 柔和边缘防止锯齿
        
        # 中心先验 (压暗四周)
        Y, X = np.ogrid[:H, :W]
        center_y, center_x = H / 2, W / 2
        sigma_y, sigma_x = H / 2.5, W / 2.5 
        center_prior = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) + (Y - center_y)**2 / (2 * sigma_y**2)))
        
        saliency_map = saliency_map * center_prior
        saliency_map = cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # ==========================================
        # 2. Felzenszwalb 超像素 (切得更细碎)
        # ==========================================
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # [核心修改 2] scale 从 100 降到 50，min_size 降到 20。
        # 块切得越小，边缘的贴合度就越高，不会被背景拖累
        segments = felzenszwalb(rgb_img, scale=50, sigma=0.5, min_size=20)
        num_nodes = np.max(segments) + 1
        
        # 背景色彩对比度先验
        boundary_mask = np.zeros((H, W), dtype=bool)
        boundary_mask[0:5, :] = True
        boundary_mask[-5:, :] = True
        boundary_mask[:, 0:5] = True
        boundary_mask[:, -5:] = True
        boundary_segments = np.unique(segments[boundary_mask])
        bg_mean_color = np.mean(rgb_img[boundary_mask], axis=0) 
        
        # 计算节点显著性
        node_saliency = np.zeros(num_nodes)
        for i in range(num_nodes):
            mask = (segments == i)
            sr_val = np.mean(saliency_map[mask])
            node_color = np.mean(rgb_img[mask], axis=0)
            color_dist = np.linalg.norm(node_color - bg_mean_color)
            node_saliency[i] = sr_val * color_dist
            
        node_saliency = (node_saliency - np.min(node_saliency)) / (np.max(node_saliency) - np.min(node_saliency) + 1e-8)
        
        saliency_2d = np.zeros((H, W), dtype=np.float32)
        for i in range(num_nodes):
            saliency_2d[segments == i] = node_saliency[i]
            
        sal_uint8 = (saliency_2d * 255).astype(np.uint8)
        otsu_thresh_val, _ = cv2.threshold(sal_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_thresh = otsu_thresh_val / 255.0
        
        # ==========================================
        # 3. 构建图割 (Graph Cut)
        # ==========================================
        rag = graph.rag_mean_color(rgb_img, segments)
        g = maxflow.Graph[float]()
        nodes = g.add_nodes(num_nodes)
        
        for i in range(num_nodes):
            sal_val = node_saliency[i]
            
            if i in boundary_segments:
                weight_fg, weight_bg = 0.0, 1000.0
            # [核心修改 3] 放宽前景门槛 (0.8)，收紧背景门槛 (0.4)
            elif sal_val > otsu_thresh * 0.8:
                weight_fg, weight_bg = 1000.0, 0.0
            elif sal_val < otsu_thresh * 0.4:
                weight_fg, weight_bg = 0.0, 1000.0
            else:
                # [核心修改 4] 气球膨胀力 (Balloon Force)
                # 强行给前景加 5.0 的保底分，鼓励向外扩张
                weight_fg = sal_val * 10.0 + 7.0
                weight_bg = (1.0 - sal_val) * 10.0
                
            g.add_tedge(nodes[i], weight_fg, weight_bg)
            
        # 添加平滑项边
        gamma = 900.0 
        # [核心修改 5] 降低平滑惩罚 (从 1.0 降至 0.3)
        # 告诉算法：别怕切出来的边缘太长、弯曲，大胆去贴合物体轮廓！
        lambda_smooth = 0.3 
        
        for edge in rag.edges:
            n1, n2 = edge
            color_diff = rag[n1][n2]['weight'] 
            smooth_weight = lambda_smooth * np.exp(- (color_diff ** 2) / gamma)
            g.add_edge(nodes[n1], nodes[n2], smooth_weight, smooth_weight)
            
        g.maxflow()
        binary_mask = np.zeros_like(segments, dtype=np.uint8)
        for i in range(num_nodes):
            if g.get_segment(nodes[i]) == 0: 
                binary_mask[segments == i] = 1
                
        # ==========================================
        # 6. 后处理：你的新主意落地 (异色区域强行挽回)
        # ==========================================
        
        # (1) 闭运算 (Close)：解决“边缘阴影或细碎部分脱离主体”的问题
        # 用一个稍微大一点的核 (比如 9x9)，把离得近的前景碎块强制“桥接”起来
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, close_kernel)
        
        # (2) 轮廓孔洞填充 (Hole Filling)：解决“眼睛、嘴巴、内部阴影被掏空”的问题
        # 寻找当前掩膜的【所有外部轮廓】
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 顺着最外围的轮廓，把里面的区域全部涂满 (thickness=cv2.FILLED)
        # 这就相当于你说的“把小区域内其他的联通块强行加到前景中去”
        cv2.drawContours(binary_mask, contours, -1, 1, thickness=cv2.FILLED)
        
        return binary_mask
    

    # # -----------------------------------#
    # # V1.0 版本的 segment() 函数，保留在这里以供对比和回滚
    # # -----------------------------------#

    # def segment(self, image):
    #     """
    #     执行完整的前景分割 (融合频域显著性、中心先验与色彩对比度先验)
    #     """
    #     H, W = image.shape[:2]
        
    #     # ==========================================
    #     # 1. 频域显著性 + 中心先验 (Center Prior)
    #     # ==========================================
    #     saliency_map = self.spectral_residual_saliency(image)
        
    #     # 生成二维中心高斯遮罩
    #     Y, X = np.ogrid[:H, :W]
    #     center_y, center_x = H / 2, W / 2
    #     sigma_y, sigma_x = H / 2.5, W / 2.5 
    #     center_prior = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) + (Y - center_y)**2 / (2 * sigma_y**2)))
        
    #     # 压暗四周，突出中心
    #     saliency_map = saliency_map * center_prior
    #     saliency_map = cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX)
        
    #     # ==========================================
    #     # 2. Felzenszwalb 超像素
    #     # ==========================================
    #     rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     segments = felzenszwalb(rgb_img, scale=self.fz_scale, sigma=self.fz_sigma, min_size=self.fz_min_size)
    #     num_nodes = np.max(segments) + 1
        
    #     # ==========================================
    #     # 3. 提取背景色彩对比度先验 (Background Contrast)
    #     # ==========================================
    #     # 提取图像四周一圈 (5 pixels) 作为绝对背景基准
    #     boundary_mask = np.zeros((H, W), dtype=bool)
    #     boundary_mask[0:5, :] = True
    #     boundary_mask[-5:, :] = True
    #     boundary_mask[:, 0:5] = True
    #     boundary_mask[:, -5:] = True
        
    #     boundary_segments = np.unique(segments[boundary_mask])
    #     bg_mean_color = np.mean(rgb_img[boundary_mask], axis=0) # 提取背景的平均 RGB 颜色
        
    #     # ==========================================
    #     # 4. 计算综合节点显著性 (Data Term 强化)
    #     # ==========================================
    #     node_saliency = np.zeros(num_nodes)
    #     for i in range(num_nodes):
    #         mask = (segments == i)
    #         # 频域显著性均值
    #         sr_val = np.mean(saliency_map[mask])
            
    #         # 计算该块颜色与背景平均颜色的欧几里得距离
    #         node_color = np.mean(rgb_img[mask], axis=0)
    #         color_dist = np.linalg.norm(node_color - bg_mean_color)
            
    #         # [核心突破] 频域(看纹理) * 色彩对比度(看颜色)
    #         # 这能完美过滤掉高频噪点(如草地)：即使草地纹理复杂(sr_val高)，
    #         # 但只要它颜色跟边缘草地一样(color_dist低)，综合显著性就会被压下去。
    #         node_saliency[i] = sr_val * color_dist
            
    #     # 综合显著性归一化到 [0, 1]
    #     node_saliency = (node_saliency - np.min(node_saliency)) / (np.max(node_saliency) - np.min(node_saliency) + 1e-8)
        
    #     # 将一维的节点显著性重建为二维图像，以便使用 Otsu 寻找全局阈值
    #     saliency_2d = np.zeros((H, W), dtype=np.float32)
    #     for i in range(num_nodes):
    #         saliency_2d[segments == i] = node_saliency[i]
            
    #     sal_uint8 = (saliency_2d * 255).astype(np.uint8)
    #     otsu_thresh_val, _ = cv2.threshold(sal_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     otsu_thresh = otsu_thresh_val / 255.0
        
    #     # ==========================================
    #     # 5. 构建图割 (Graph Cut)
    #     # ==========================================
    #     rag = graph.rag_mean_color(rgb_img, segments)
    #     g = maxflow.Graph[float]()
    #     nodes = g.add_nodes(num_nodes)
        
    #     # 添加数据项边
    #     for i in range(num_nodes):
    #         sal_val = node_saliency[i]
            
    #         if i in boundary_segments:
    #             # 依然保留边界背景先验，稳住阵脚
    #             weight_fg, weight_bg = 0.0, 1000.0
    #         elif sal_val > otsu_thresh * 1.1:
    #             weight_fg, weight_bg = 1000.0, 0.0
    #         elif sal_val < otsu_thresh * 0.8:
    #             weight_fg, weight_bg = 0.0, 1000.0
    #         else:
    #             weight_fg = sal_val * 10.0
    #             weight_bg = (1.0 - sal_val) * 10.0
                
    #         g.add_tedge(nodes[i], weight_fg, weight_bg)
            
    #     # 添加平滑项边
    #     gamma = 900.0 
    #     lambda_smooth = 1.0 
    #     for edge in rag.edges:
    #         n1, n2 = edge
    #         color_diff = rag[n1][n2]['weight'] 
    #         smooth_weight = lambda_smooth * np.exp(- (color_diff ** 2) / gamma)
    #         g.add_edge(nodes[n1], nodes[n2], smooth_weight, smooth_weight)
            
    #     # 求解最大流最小割
    #     g.maxflow()
    #     binary_mask = np.zeros_like(segments, dtype=np.uint8)
    #     for i in range(num_nodes):
    #         if g.get_segment(nodes[i]) == 0: 
    #             binary_mask[segments == i] = 1
                
    #     # [彻底移除连通域拦截逻辑]，直接返回原始的、允许小碎片前景的图割掩膜
    #     # return binary_mask

    #     # V2.0 版本的 ,直接添加
    #     # ==========================================
    #     # 6. 后处理：你的新主意落地 (异色区域强行挽回)
    #     # ==========================================
        
    #     # (1) 闭运算 (Close)：解决“边缘阴影或细碎部分脱离主体”的问题
    #     # 用一个稍微大一点的核 (比如 9x9)，把离得近的前景碎块强制“桥接”起来
    #     close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    #     binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, close_kernel)
        
    #     # (2) 轮廓孔洞填充 (Hole Filling)：解决“眼睛、嘴巴、内部阴影被掏空”的问题
    #     # 寻找当前掩膜的【所有外部轮廓】
    #     contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    #     # 顺着最外围的轮廓，把里面的区域全部涂满 (thickness=cv2.FILLED)
    #     # 这就相当于你说的“把小区域内其他的联通块强行加到前景中去”
    #     cv2.drawContours(binary_mask, contours, -1, 1, thickness=cv2.FILLED)
        
    #     return binary_mask