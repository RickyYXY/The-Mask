from TediGAN.generation_api import *
from Sefa.api import *
import numpy as np
import torch
import cv2


def imshow(images, col, viz_size=256):
    """Shows images in one figure."""
    num, height, width, channels = images.shape
    assert num % col == 0
    row = num // col

    fused_image = np.zeros(
        (viz_size * row, viz_size * col, channels), dtype=np.uint8)

    for idx, image in enumerate(images):
        i, j = divmod(idx, col)
        y = i * viz_size
        x = j * viz_size
        if height != viz_size or width != viz_size:
            image = cv2.resize(image, (viz_size, viz_size))
        fused_image[y:y + viz_size, x:x + viz_size] = image

    fused_image = np.asarray(fused_image, dtype=np.uint8)
    fused_image = cv2.cvtColor(fused_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("test_imgs", fused_image)
    cv2.waitKey(0)


num = 1
description = 'she, young, yellow hair'

codes, _ = ImageEdit(description=description)
# z_space_dim = 512
# seed = 0
# torch.manual_seed(seed)
# codes = torch.randn(num, z_space_dim).cuda()

layer_index = 'all'
num_semantics = 5
step = [5, 5, 5, 5, 5]

res_img = code_to_img_api(codes, layer_index, num_semantics, step)
imshow(res_img, num)
