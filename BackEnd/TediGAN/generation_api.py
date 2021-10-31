import os
import sys
#sys.path.append('TediGAN')
lswpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(lswpath, 'TediGAN'))
import numpy as np
from PIL import Image

from base.utils.inverter import StyleGANInverter
from base.utils.visualizer import resize_image

# img_file = os.path.join('base', 'examples', '142.jpg')
# desc = 'she is young'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
hyper_parameter = {
    'model_name': 'styleganinv_ffhq256',
    'step': 200,
    'lr': 0.01,
    'lambda_clip': 1.0,
    'lambda_feat': 5e-5,
    'lambda_l2': 1.0,
    'lambda_enc': 2.0,
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

    if image_path is None:
        z, viz_results = inverter.invert(None)
    else:
        image_size = inverter.G.resolution
        image = Image.open(image_path)

        image = resize_image(np.array(image), (image_size, image_size))
        z, viz_results = inverter.easy_invert(image, 1)

    return z, viz_results[-1]


def save_code(code, path=os.path.join('tmp_codes', 'code.npy')):
    path_split = os.path.split(path)
    directory = os.path.join(*path_split[:-1])
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)
    np.save(path, code)


# latant_code, fixed_image = ImageEdit(image_path=img_file, description=desc)
