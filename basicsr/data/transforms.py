import cv2
import random
import torch


def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int | list[int]): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    if isinstance(gt_patch_size, list):
        assert len(gt_patch_size) == 2, "gt_patch_size must be a list of two integers: [width, height]"
        lq_patch_size_w, lq_patch_size_h = gt_patch_size[0] // scale, gt_patch_size[1] // scale
        gt_patch_size_w, gt_patch_size_h = gt_patch_size
    elif isinstance(gt_patch_size, int):
        lq_patch_size_w = gt_patch_size // scale
        lq_patch_size_h = lq_patch_size_w
        gt_patch_size_w, gt_patch_size_h = gt_patch_size, gt_patch_size
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    # 如果图像比要crop的patch size还小，则等比例resize图像
    if h_lq < lq_patch_size_h or w_lq < lq_patch_size_w:
        ratio_w = w_lq / lq_patch_size_w
        ratio_h = h_lq / lq_patch_size_h
        if ratio_w < ratio_h:
            h_lq = int(lq_patch_size_w / w_lq * h_lq)
            w_lq = lq_patch_size_w
            img_lqs = [cv2.resize(v, (w_lq, h_lq), interpolation=cv2.INTER_CUBIC) for v in img_lqs]
        else:
            w_lq = int(lq_patch_size_h / h_lq * w_lq)
            h_lq = lq_patch_size_h
            img_lqs = [cv2.resize(v, (w_lq, h_lq), interpolation=cv2.INTER_CUBIC) for v in img_lqs]
    # 下面这种情况通常发生于scale==1，即不是超分辨率任务
    if h_gt < gt_patch_size_h or w_gt < gt_patch_size_w:
        ratio_w = w_gt / gt_patch_size_w
        ratio_h = h_gt / gt_patch_size_h
        if ratio_w < ratio_h:
            h_gt = int(gt_patch_size_w / w_gt * h_gt)
            w_gt = gt_patch_size_w
            img_gts = [cv2.resize(v, (w_gt, h_gt), interpolation=cv2.INTER_CUBIC) for v in img_gts]
        else:
            w_gt = int(gt_patch_size_h / h_gt * w_gt)
            h_gt = gt_patch_size_h
            img_gts = [cv2.resize(v, (w_gt, h_gt), interpolation=cv2.INTER_CUBIC) for v in img_gts]

    # if h_lq < lq_patch_size or w_lq < lq_patch_size:
    #     raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
    #                      f'({lq_patch_size}, {lq_patch_size}). '
    #                      f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size_h)
    left = random.randint(0, w_lq - lq_patch_size_w)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size_h, left:left + lq_patch_size_w] for v in img_lqs]
        # 如果在dataset.__getitem__中调用并且是Tensor类型，不应该是这样吗：
        #  img_lqs = [v[:, top:top + lq_patch_size_h, left:left + lq_patch_size_w] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size_h, left:left + lq_patch_size_w, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size_h, left_gt:left_gt + gt_patch_size_w] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size_h, left_gt:left_gt + gt_patch_size_w, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img
