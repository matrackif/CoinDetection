"""
Marcus D. Bloice, Christof Stocker, and Andreas Holzinger,
Augmentor: An Image Augmentation Library for Machine Learning,
arXiv preprint arXiv:1708.04680, https://arxiv.org/abs/1708.04680, 2017.
Asciicast
"""
import Augmentor

if __name__ == '__main__':
    dir_names = ['1_2_5_gr_tails', '1_gr_heads', '1_zl_heads', '2_gr_heads', '2_zl_heads',
                 '2_zl_tails',
                 '5_gr_heads', '5_zl_heads', '5_zl_tails', '10_20_50_1_tails', '10_gr_heads',
                 '20_gr_heads',
                 '50_gr_heads']
    for dirname in dir_names:
        p = Augmentor.Pipeline(dirname)
        p.rotate90(probability=0.5)
        p.rotate180(probability=0.5)
        p.rotate270(probability=0.5)
        p.shear(probability=0.5, max_shear_left=5, max_shear_right=5)
        # p.crop_random(probability=1, percentage_area=0.2)
        p.zoom(probability=0.1, min_factor=1.1, max_factor=1.5)
        p.random_distortion(probability=0.1, grid_width=4, grid_height=4, magnitude=5)
        p.resize(probability=1.0, width=150, height=150)
        p.skew_tilt(probability=0.1)
        p.skew_corner(probability=0.1)
        p.sample(300)
