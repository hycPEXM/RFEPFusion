import numpy as np
import torch

from basicsr.utils.registry import METRIC_REGISTRY

__all__ = ['SegmentationMetric', 'calculate_mIoUD', 'calculate_mIoUI', 
           'VIF_function', 'EN_function', 'AG_function']

"""
confusionMetric 矩阵中(i, j)位置的元素代表该张图片中真实类别为i,被预测为j的像素个数；行为label，列为prediction
L\P    P    N
P      TP    FN
N      FP    TN
输入的label和predict皆为Pytorch张量
"""
class SegmentationMetric(object):
    def __init__(self, numClass, device):
        self.numClass = numClass
        self.confusionMatrix = torch.zeros((self.numClass,) * 2).to(device)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        # acc = acc.item()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        # classAcc  = classAcc.item()
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = classAcc[classAcc < float('inf')].mean() # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        # meanAcc = meanAcc.item()
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = torch.sum(self.confusionMatrix, axis=1) + torch.sum(self.confusionMatrix, axis=0) - torch.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        # IoU = [a.item() for a in IoU] 
        return IoU

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        mIoU = IoU[IoU<float('inf')].mean()# 求各类别IoU的平均
        mIoU = mIoU.item()  # 返回数值！
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel, ignore_labels):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        for IgLabel in ignore_labels:
            mask &= (imgLabel != IgLabel)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = torch.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.view(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU, 频权交并比:为MIoU的一种提升, 这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = torch.sum(self.confusion_matrix, axis=1) / torch.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                torch.sum(self.confusion_matrix, axis=1) + torch.sum(self.confusion_matrix, axis=0) -
                torch.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        # FWIoU = FWIoU.item()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel, ignore_labels):
        assert imgPredict.shape == imgLabel.shape
        with torch.no_grad():
            self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel, ignore_labels)  # 得到混淆矩阵
        return self.confusionMatrix

    @torch.no_grad()
    def per_image_mIoU(self, imgPredict, imgLabel, ignore_labels):
        assert imgPredict.shape == imgLabel.shape
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        for IgLabel in ignore_labels:
            mask &= (imgLabel != IgLabel)
        imgLabel = imgLabel[mask]
        not_null_label = torch.unique(imgLabel, sorted=True).tolist()  # active classes
        if len(not_null_label) == 0:  # 无效的label图（全是ignore？）
            return 0
        confusionMat = self.genConfusionMatrix(imgPredict, imgLabel, ignore_labels)
        iou = 0
        for i in not_null_label:
            tp = confusionMat[i, i].item()
            fp = confusionMat[:, i].sum().item() - tp
            fn = confusionMat[i, :].sum().item() - tp
            denominator = tp + fp + fn
            if denominator == 0:
                iou += 0.0  # 避免除以零
            else:
                iou += tp / denominator
        return iou/len(not_null_label)

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))

@METRIC_REGISTRY.register()
def calculate_mIoUD(seg_result, label, nc=9, labels_to_ignore=[255], device='cpu', **kwargs):
    if not hasattr(calculate_mIoUD, "seg_metric"):    
        calculate_mIoUD.seg_metric = SegmentationMetric(numClass=nc, device=device)
    label = label.to(device)
    seg_result = seg_result.to(device)
    # seg_result = torch.argmax(seg_result, dim=1, keepdim=True)
    calculate_mIoUD.seg_metric.addBatch(seg_result, label, labels_to_ignore)
    return calculate_mIoUD.seg_metric

@METRIC_REGISTRY.register()
def calculate_mIoUI(seg_result, label, nc=9, labels_to_ignore=[255], device='cpu', **kwargs):
    if not hasattr(calculate_mIoUI, "seg_metric"):    
        calculate_mIoUI.seg_metric = SegmentationMetric(numClass=nc, device=device)
    label = label.to(device)
    seg_result = seg_result.to(device)
    # seg_result = torch.argmax(seg_result, dim=1, keepdim=True)
    return calculate_mIoUI.seg_metric.per_image_mIoU(seg_result, label, labels_to_ignore)

# fusion 

from scipy.signal import convolve2d

def fspecial_gaussian(shape, sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',...)
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# dist: distorted image
def vifp_mscale(ref, dist):
    sigma_nsq = 2
    num = 0
    den = 0
    for scale in range(1, 5):
        N = 2**(4-scale+1)+1
        win = fspecial_gaussian((N, N), N/5)

        if scale > 1:
            ref = convolve2d(ref, win, mode='valid')
            dist = convolve2d(dist, win, mode='valid')
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = convolve2d(ref, win, mode='valid')
        mu2 = convolve2d(dist, win, mode='valid')
        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2
        sigma1_sq = convolve2d(ref*ref, win, mode='valid') - mu1_sq
        sigma2_sq = convolve2d(dist*dist, win, mode='valid') - mu2_sq
        sigma12 = convolve2d(ref*dist, win, mode='valid') - mu1_mu2
        sigma1_sq[sigma1_sq<0] = 0
        sigma2_sq[sigma2_sq<0] = 0

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g*sigma12

        g[sigma1_sq<1e-10] = 0
        sv_sq[sigma1_sq<1e-10] = sigma2_sq[sigma1_sq<1e-10]
        sigma1_sq[sigma1_sq<1e-10] = 0

        g[sigma2_sq<1e-10] = 0
        sv_sq[sigma2_sq<1e-10] = 0

        sv_sq[g<0] = sigma2_sq[g<0]
        g[g<0] = 0
        sv_sq[sv_sq<=1e-10] = 1e-10

        num += np.sum(np.log10(1+g**2 * sigma1_sq/(sv_sq+sigma_nsq)))
        den += np.sum(np.log10(1+sigma1_sq/sigma_nsq))
    vifp = num/den
    return vifp

@METRIC_REGISTRY.register()
def VIF_function(F, A, B, **kwargs):
    VIF = vifp_mscale(A, F) + vifp_mscale(B, F)
    return VIF

@METRIC_REGISTRY.register()
def EN_function(F, **kwargs):
    # 计算图像的直方图
    F = F.astype(np.int32)
    histogram, bins = np.histogram(F, bins=256, range=(0, 255))
    # 将直方图归一化
    histogram = histogram / float(np.sum(histogram))
    # 计算熵
    entropy = -np.sum(histogram * np.log2(histogram + 1e-9))
    return entropy

@METRIC_REGISTRY.register()
def AG_function(F, **kwargs):
    # can't write as this way: np.array(F), otherwise it will raise the error like:
    # TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    # if isinstance(F, torch.Tensor):
    #     F = F.cpu().numpy()
    # else:
    #     F = np.array(F)
    width = F.shape[1]
    width = width - 1
    height = F.shape[0]
    height = height - 1
    # tmp = 0.0
    [grady, gradx] = np.gradient(F)
    s = np.sqrt((np.square(gradx) + np.square(grady)) / 2)
    AG = np.sum(s) / (width * height)
    return AG

@METRIC_REGISTRY.register()
def SF_function(F, **kwargs):
    # F = np.array(F)
    RF = np.diff(F, axis=0)
    RF1 = np.sqrt(np.mean(np.mean(RF ** 2)))
    CF = np.diff(F, axis=1)
    CF1 = np.sqrt(np.mean(np.mean(CF ** 2)))
    SF = np.sqrt(RF1 ** 2 + CF1 ** 2)
    return SF