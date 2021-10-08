import os
import sys
sys.path.insert(0, "base")

import clip
import numpy as np
import torch
from PIL import Image

from base.utils.inverter import StyleGANInverter
from base.utils.visualizer import resize_image

# img_file = os.path.join('base', 'examples', '142.jpg')
# desc = 'she is young'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
hyper_parameter = {
    'model_name': 'styleganinv_ffhq256',
    'step': 3,
    'lr': 0.01,
    'lambda_clip': 1.0,
    'lambda_feat': 2.0,
    'lambda_l2': 1.0,
    'lambda_enc': 5,
}


def ImageEdit(image_path=None, description=None):
    mode = 'gen' if image_path is None else 'man'
    if not description and not image_path:
        return None

    inverter = StyleGANInverter(hyper_parameter['model_name'],
                                mode=mode,
                                learning_rate=hyper_parameter['lr'],
                                iteration=hyper_parameter['step'],
                                reconstruction_loss_weight=hyper_parameter['lambda_l2'],
                                perceptual_loss_weight=hyper_parameter['lambda_feat'],
                                regularization_loss_weight=hyper_parameter['lambda_enc'],
                                clip_loss_weight=hyper_parameter['lambda_clip'],
                                description=description)

    if description is None:
        image = inverter.preprocess(image_path)
        return inverter.get_init_code(image), image

    image_size = inverter.G.resolution

    text_inputs = torch.cat([clip.tokenize(description)]).cpu()
    # Invert images.
    # uploaded_file = uploaded_file.read()
    image = Image.open(image_path)
    image = resize_image(np.array(image), (image_size, image_size))
    z, viz_results = inverter.easy_invert(image, 1)

    return z, viz_results[-1]


# latant_code, fixed_image = ImageEdit(image_path=img_file, description=desc)

