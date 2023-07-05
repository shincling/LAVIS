'''
Author: Diantao Tu
Date: 2023-04-15 20:51:25
'''
from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import scipy.ndimage
import scipy.interpolate
import scipy.stats
import numpy as np
import torch
from collections.abc import Sequence, Mapping
import copy
import random
import logging



class Copy(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict(coord="origin_coord", label="origin_label")
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[value] = data_dict[key].copy()
            elif isinstance(data_dict[key], torch.Tensor):
                data_dict[value] = data_dict[key].clone().detach()
            else:
                data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict

class RandomSample(object):
    def __init__(self, max_num = 80000):
        self.max_num = max_num
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            if data_dict["coord"].shape[0] > self.max_num:
                choice = np.random.choice(data_dict["coord"].shape[0], self.max_num, replace=False)
                for key in data_dict.keys():
                    data_dict[key] = data_dict[key][choice]
            else:
                #若不够这么多点，则随机的重复采样
                choice = np.random.choice(data_dict["coord"].shape[0], self.max_num - data_dict["coord"].shape[0], replace=True)
                for key in data_dict.keys():
                    data_dict[key] = np.concatenate([data_dict[key], data_dict[key][choice]], axis=0)
        return data_dict

class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            x_min, y_min, z_min = data_dict["coord"].min(axis=0)
            x_max, y_max, _ = data_dict["coord"].max(axis=0)
            if self.apply_z:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
            else:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
            data_dict["coord"] -= shift
        return data_dict

class RandomScale(object):
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
            data_dict["coord"] *= scale
        return data_dict
    
class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]
        return data_dict

class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert (clip > 0)
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.clip(self.sigma * np.random.randn(data_dict["coord"].shape[0], 3), -self.clip, self.clip)
            data_dict["coord"] += jitter
        return data_dict

class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            lo = np.min(data_dict["color"], 0, keepdims=True)
            hi = np.max(data_dict["color"], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data_dict["color"][:, :3] - lo) * scale
            blend_factor = np.random.rand() if self.blend_factor is None else self.blend_factor
            data_dict["color"][:, :3] = (1 - blend_factor) * data_dict["color"][:, :3] + blend_factor * contrast_feat
        return data_dict
    
class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data_dict["color"][:, :3] = np.clip(tr + data_dict["color"][:, :3], 0, 255)
        return data_dict

class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            noise = np.random.randn(data_dict["color"].shape[0], 3)
            noise *= self.std * 255
            data_dict["color"][:, :3] = np.clip(noise + data_dict["color"][:, :3], 0, 255)
        return data_dict

class Voxelize(object):
    def __init__(self,
                 voxel_size=0.05,
                 hash_type="fnv",
                 mode='train',
                 keys=("coord", "normal", "color", "label"),
                 return_inverse=False,
                 return_discrete_coord=False,
                 return_min_coord=False):
        self.voxel_size = voxel_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_inverse = return_inverse
        self.return_discrete_coord = return_discrete_coord
        self.return_min_coord = return_min_coord

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        discrete_coord = np.floor(data_dict["coord"] / np.array(self.voxel_size)).astype(np.int)
        min_coord = discrete_coord.min(0) * np.array(self.voxel_size)
        discrete_coord -= discrete_coord.min(0)
        key = self.hash(discrete_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == 'train':  # train mode
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_unique = idx_sort[idx_select]
            if self.return_discrete_coord:
                data_dict["discrete_coord"] = discrete_coord[idx_unique]
            if self.return_inverse:
                data_dict["mask"] = np.zeros_like(inverse)
                data_dict["mask"][idx_unique] = 1
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
                data_dict["length"] = np.array(inverse.shape)
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == 'test':  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                # TODO to be more robust
                for key in self.keys:
                    data_part[key] = data_dict[key][idx_part]
                if self.return_discrete_coord:
                    data_part["discrete_coord"] = discrete_coord[idx_part]
                if self.return_inverse:
                    data_part["inverse"] = np.zeros_like(inverse)
                    data_part["inverse"][idx_sort] = inverse
                    data_part["length"] = np.array(inverse.shape)
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

class SphereCrop(object):
    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all"]
        self.mode = mode

    def __call__(self, data_dict):
        point_max = int(self.sample_rate * data_dict["coord"].shape[0]) \
            if self.sample_rate is not None else self.point_max

        assert "coord" in data_dict.keys()
        if self.mode == "all":
            # TODO: Optimize
            if "index" not in data_dict.keys():
                data_dict["index"] = np.arange(data_dict["coord"].shape[0])
            data_part_list = []
            # coord_list, color_list, dist2_list, idx_list, offset_list = [], [], [], [], []
            if data_dict["coord"].shape[0] > point_max:
                coord_p, idx_uni = np.random.rand(data_dict["coord"].shape[0]) * 1e-3, np.array([])
                while idx_uni.size != data_dict["index"].shape[0]:
                    init_idx = np.argmin(coord_p)
                    dist2 = np.sum(np.power(data_dict["coord"] - data_dict["coord"][init_idx], 2), 1)
                    idx_crop = np.argsort(dist2)[:point_max]

                    data_crop_dict = dict()
                    if "coord" in data_dict.keys():
                        data_crop_dict["coord"] = data_dict["coord"][idx_crop]
                    if "discrete_coord" in data_dict.keys():
                        data_crop_dict["discrete_coord"] = data_dict["discrete_coord"][idx_crop]
                    if "normal" in data_dict.keys():
                        data_crop_dict["normal"] = data_dict["normal"][idx_crop]
                    if "color" in data_dict.keys():
                        data_crop_dict["color"] = data_dict["color"][idx_crop]
                    data_crop_dict["weight"] = dist2[idx_crop]
                    data_crop_dict["index"] = data_dict["index"][idx_crop]
                    data_part_list.append(data_crop_dict)

                    delta = np.square(1 - data_crop_dict["weight"] / np.max(data_crop_dict["weight"]))
                    coord_p[idx_crop] += delta
                    idx_uni = np.unique(np.concatenate((idx_uni, data_crop_dict["index"])))
            else:
                data_crop_dict = data_dict.copy()
                data_crop_dict["weight"] = np.zeros(data_dict["coord"].shape[0])
                data_crop_dict["index"] = data_dict["index"]
                data_part_list.append(data_crop_dict)
            return data_part_list
        # mode is "random" or "center"
        elif data_dict["coord"].shape[0] > point_max:
            if self.mode == "random":
                center = data_dict["coord"][np.random.randint(data_dict["coord"].shape[0])]
            elif self.mode == "center":
                center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[:point_max]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx_crop]
            if "origin_coord" in data_dict.keys():
                data_dict["origin_coord"] = data_dict["origin_coord"][idx_crop]
            if "discrete_coord" in data_dict.keys():
                data_dict["discrete_coord"] = data_dict["discrete_coord"][idx_crop]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx_crop]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx_crop]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx_crop]
            if "label" in data_dict.keys():
                data_dict["label"] = data_dict["label"][idx_crop] \
                    if len(data_dict["label"]) != 1 else data_dict["label"]
        return data_dict

class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            x_min, y_min, z_min = data_dict["coord"].min(axis=0)
            x_max, y_max, _ = data_dict["coord"].max(axis=0)
            if self.apply_z:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
            else:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
            data_dict["coord"] -= shift
        return data_dict

class NormalizeColor(object):
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"] / 127.5 - 1
        return data_dict

class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        elif isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.int):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        else:
            raise TypeError(f'type {type(data)} cannot be converted to tensor.')

class Collect(object):
    def __init__(self,
                 keys,
                 offset_keys_dict=None,
                 **kwargs
                 ):
        """
            e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data

class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
            upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx]
            if "label" in data_dict.keys():
                data_dict["label"] = data_dict["label"][idx] \
                    if len(data_dict["label"]) != 1 else data_dict["label"]
        return data_dict

class RandomRotate(object):
    def __init__(self,
                 angle=None,
                 center=None,
                 axis='z',
                 always_apply=False,
                 p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == 'x':
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == 'y':
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == 'z':
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict
    
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                       (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=False, fill_value=0)
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["coord"] = self.elastic_distortion(data_dict["coord"], granularity, magnitude)
        return data_dict

@registry.register_processor("chinese_caption")
class ChineseCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        # super().__init__()
        self.prompt = prompt            # 对每个caption添加的前缀
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption, self.max_words)

        return caption
    
    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)
    
    def pre_caption(self, caption, max_words=50):
        # TODO 对中文caption进行预处理
        # 把开头、结尾的空格去掉
        caption = caption.strip()
        # 把回车符号删掉
        caption = caption.replace("\r", "")
        # 把中间的空格都去掉
        caption = caption.replace(" ", "")
        # 如果超出最大长度, 则截断
        if len(caption) > max_words:
            caption = caption[:max_words]

        return caption


@registry.register_processor("cloud_train")
class CloudTrainProcessor(BaseProcessor):
    def __init__(self, max_size:int):
        super().__init__()
        self.max_size = max_size
        logging.info("cloud max size: {}".format(self.max_size))

    # TODO 对点云进行裁剪, 降采样, 数据增强等等
    def __call__(self, point_cloud):

        coord = point_cloud["coord"]
        color = point_cloud["color"]
        if "semantic_gt" in point_cloud.keys():
            label = point_cloud["semantic_gt"].reshape(-1)
        else:
            label = np.zeros(coord.shape[0])
        pcd_dict = dict(coord=coord, color=color, label=label)

         # 接下来是对点云进行数据增强
        # 随机下采样是我自己实现的，而对于其他的数据变换方式是原来的PointTransformer2的代码
        pcd_dict = RandomSample(max_num=self.max_size)(pcd_dict)        
        pcd_dict = CenterShift(apply_z=True)(pcd_dict)
        # pcd_dict = RandomDropout(dropout_ratio=0.2, dropout_application_ratio=0.2)(pcd_dict)
        pcd_dict = RandomRotate(angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5)(pcd_dict)
        pcd_dict = RandomRotate(angle=[-1/64, 1/64], axis='x', p=0.5)(pcd_dict)
        pcd_dict = RandomRotate(angle=[-1/64, 1/64], axis='y', p=0.5)(pcd_dict)
        pcd_dict = RandomScale(scale=[0.9, 1.1])(pcd_dict)
        pcd_dict = RandomFlip(p=0.5)(pcd_dict)
        pcd_dict = RandomJitter(sigma=0.005, clip=0.02)(pcd_dict)
        # pcd_dict = ElasticDistortion(distortion_params=[[0.2, 0.4], [0.8, 1.6]])(pcd_dict)
        pcd_dict = ChromaticAutoContrast(p=0.2, blend_factor=None)(pcd_dict)
        pcd_dict = ChromaticTranslation(p=0.95, ratio=0.05)(pcd_dict)
        pcd_dict = ChromaticJitter(p=0.95, std=0.05)(pcd_dict)        
        pcd_dict = CenterShift(apply_z=False)(pcd_dict)
        pcd_dict = NormalizeColor()(pcd_dict)
        pcd_dict = ToTensor()(pcd_dict)
        pcd_dict = Collect(keys=("coord", "color", "label"), feat_keys=["coord", "color"])(pcd_dict)

        return pcd_dict 
    
    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        max_size = cfg.get("max_size", 2048)

        return cls(max_size=max_size)
    

@registry.register_processor("cloud_test")
class CloudTestProcessor(BaseProcessor):
    def __init__(self, max_size:int):
        super().__init__()
        self.max_size = max_size

    # TODO 对点云进行裁剪, 降采样等等
    def __call__(self, point_cloud):

        coord = point_cloud["coord"]
        color = point_cloud["color"]
        if "semantic_gt" in point_cloud.keys():
            label = point_cloud["semantic_gt"].reshape(-1)
        else:
            label = np.zeros(coord.shape[0])
        pcd_dict = dict(coord=coord, color=color, label=label)

        # 接下来是对点云进行数据增强
        pcd_dict = RandomSample(max_num=self.max_size)(pcd_dict)        
        pcd_dict = CenterShift(apply_z=True)(pcd_dict)
        pcd_dict = NormalizeColor()(pcd_dict)
        pcd_dict = ToTensor()(pcd_dict)
        pcd_dict = Collect(keys=("coord", "color", "label"), feat_keys=["coord", "color"])(pcd_dict)
        return pcd_dict 
    
    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        max_size = cfg.get("max_size", 2048)

        return cls(max_size=max_size)
    