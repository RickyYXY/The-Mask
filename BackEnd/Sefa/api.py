"""SeFa."""

import os
import argparse
from tqdm import tqdm
import numpy as np
import sys
import torch
sys.path.append('Sefa')
from models import parse_gan_type
from utils import to_tensor
from utils import postprocess
from utils import load_generator
from utils import factorize_weight


def synthesize(generator, gan_type, codes, sefa_usage=None):
    """Synthesizes images with the give codes."""
    if gan_type == 'pggan':
        images = generator(to_tensor(codes))['image']
    elif gan_type in ['stylegan', 'stylegan2']:
        images = generator.synthesis(to_tensor(codes))['image']
    elif gan_type == 'stylegan_inv':
        # 默认为z_type
        images = generator.synthesize(
            codes, sefa_usage, latent_space_type='z')['image']
    images = postprocess(images)
    return images


def sample(generator, gan_type, codes):
    """Samples latent codes."""
    if gan_type == 'pggan':
        codes = generator.layer0.pixel_norm(codes)
    elif gan_type == 'stylegan':
        codes = generator.mapping(codes)['w']
        codes = generator.truncation(codes, trunc_psi=0.7, trunc_layers=8)
    elif gan_type == 'stylegan2':
        codes = generator.mapping(codes)['w']
        codes = generator.truncation(codes, trunc_psi=0.5, trunc_layers=18)
    codes = codes.detach().cpu().numpy()
    return codes


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

    if gan_type != "stylegan_inv":
        temp_codes = sample(generator, gan_type, codes)
        for sem_id in tqdm(range(num_sem), desc='Semantic ', leave=False):
            boundary = boundaries[sem_id:sem_id + 1]
            d = distances[step[sem_id]]
            if gan_type == 'pggan':
                temp_codes += boundary * d
            elif gan_type in ['stylegan', 'stylegan2']:
                temp_codes[:, layers, :] += boundary * d
        new_images = synthesize(generator, gan_type, temp_codes)
    else:
        temp_codes = codes.detach().cpu().numpy()
        sefa_usage = {
            'step': step,
            'num_sem': num_sem,
            'distances': distances,
            'boundaries': boundaries,
            'layers': layers
        }
        new_images = synthesize(generator, gan_type, temp_codes, sefa_usage)
    return new_images
