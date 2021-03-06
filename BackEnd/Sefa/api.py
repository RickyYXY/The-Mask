"""SeFa."""
import sys, os
# sys.path.append('Sefa')
lswpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(sys.path)
sys.path.insert(0, os.path.join(lswpath, 'Sefa'))
from models_sefa import parse_gan_type
from utils_sefa import to_tensor
from utils_sefa import postprocess
from utils_sefa import load_generator
from utils_sefa import factorize_weight
import torch
import numpy as np
from tqdm import tqdm
import argparse



def synthesize(generator, gan_type, codes):
    """Synthesizes images with the give codes."""
    if gan_type == 'pggan':
        images = generator(to_tensor(codes))['image']
    elif gan_type in ['stylegan', 'stylegan2']:
        images = generator.synthesis(to_tensor(codes))['image']
    elif gan_type == 'stylegan_inv':
        # 默认为z_type
        images = generator.synthesize(
            codes, latent_space_type='wp')['image']
    images = postprocess(images)
    return images


def sample(generator, gan_type, codes):
    """Samples latent codes.
        stylegan_inv don't need this"""
    if gan_type == 'pggan':
        codes = generator.layer0.pixel_norm(codes)
    elif gan_type == 'stylegan':
        codes = generator.mapping(codes)['w']
        codes = generator.truncation(codes, trunc_psi=0.7, trunc_layers=8)
    elif gan_type == 'stylegan2':
        codes = generator.mapping(codes)['w']
        codes = generator.truncation(codes, trunc_psi=0.5, trunc_layers=18)
    if not isinstance(codes, np.ndarray):
        codes = codes.detach().cpu().numpy()
    return codes


def load_code(code_dir):
    code = np.load(code_dir)
    return code


def code_to_img_api(codes, layer_idx, num_semantics, step,
                    start_distance=-3, end_distance=3, step_num=11, model_name='styleganinv_ffhq256_generator'):
    """
    code: code for img
    layer_idx: 'all', '0-1', '2-5', '6-13'
    num_semantics: count of the chosen semantics
    step: List, the value indexs of distance(0~10), depend on step_num
    return: numpy imgs N*H*W*C
    """
    generator = load_generator(model_name)
    gan_type = parse_gan_type(generator)
    layers, boundaries, values = factorize_weight(generator, layer_idx)

    distances = np.linspace(start_distance, end_distance, step_num)
    num_sem = num_semantics

    temp_codes = sample(generator, gan_type, codes)
    for sem_id in tqdm(range(num_sem), desc='Semantic ', leave=False):
        boundary = boundaries[sem_id:sem_id + 1]
        d = distances[step[sem_id]]
        if gan_type == 'pggan':
            temp_codes += boundary * d
        elif gan_type in ['stylegan', 'stylegan2', 'stylegan_inv']:
            temp_codes[:, layers, :] += boundary * d
    new_images = synthesize(generator, gan_type, temp_codes)

    return new_images
