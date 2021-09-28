from torch.random import seed
from api import *
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
    cv2.imshow("test_imgs", fused_image)
    cv2.waitKey(0)
    # data = io.BytesIO()
    # PIL.Image.fromarray(fused_image).save(data, 'jpeg')
    # im_data = data.getvalue()
    # disp = IPython.display.display(IPython.display.Image(im_data))
    # return disp


seed = 0
num = 1
z_space_dim = 512
torch.manual_seed(seed)

codes = torch.randn(num, z_space_dim).cuda()
layer_index = 'all'
num_semantics = 5
step = [5, 5, 5, 5, 5]

res_img = code_to_img_api(codes, layer_index, num_semantics, step)
imshow(res_img, num)
