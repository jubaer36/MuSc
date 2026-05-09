import os
import json
from enum import Enum
import PIL
import torch
from torchvision import transforms
import random

_CLASSNAMES = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MVTecAD2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TEST,
        clip_transformer=None,
        k_shot=0,
        random_seed=42,
        divide_num=1,
        divide_iter=0,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES

        self.data_to_iterate = self.get_image_data()
        if divide_num > 1:
            self.data_to_iterate = self.sub_datasets(self.data_to_iterate, divide_num, divide_iter, random_seed)

        if k_shot > 0:
            torch.manual_seed(random_seed)
            if k_shot < len(self.data_to_iterate):
                indices = torch.randint(0, len(self.data_to_iterate), (k_shot,))
                self.data_to_iterate = [self.data_to_iterate[i] for i in indices]

        if clip_transformer is None:
            self.transform_img = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform_img = clip_transformer

        self.transform_mask = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ])

        self.imagesize = (3, imagesize, imagesize)

    def sub_datasets(self, full_datasets, divide_num, divide_iter, random_seed=42):
        if divide_num == 0:
            return full_datasets
        random.seed(random_seed)

        id_dict = {}
        for i in range(len(full_datasets)):
            # full_datasets[i][2] is img_path; parent dir is 'good' or 'bad'
            anomaly_type = full_datasets[i][2].split('/')[-2]
            if anomaly_type not in id_dict:
                id_dict[anomaly_type] = []
            id_dict[anomaly_type].append(i)

        sub_id_list = []
        for k in id_dict.keys():
            type_id_list = id_dict[k]
            random.shuffle(type_id_list)
            devide_list = [type_id_list[i:i+divide_num] for i in range(0, len(type_id_list), divide_num)]
            sub_list = [devide_list[i][divide_iter] for i in range(len(devide_list)) if len(devide_list[i]) > divide_iter]
            sub_id_list.extend(sub_list)

        return [full_datasets[id] for id in sub_id_list]

    def __getitem__(self, idx):
        classname, specie_name, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "is_anomaly": int(specie_name != "good"),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        meta_path = os.path.join(self.source, "meta_mvtec2.json")
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        test_meta = meta['test']
        data_to_iterate = []

        for classname in sorted(self.classnames_to_use):
            if classname not in test_meta:
                continue
            entries = sorted(test_meta[classname], key=lambda x: x['img_path'])
            for entry in entries:
                img_path = os.path.join(self.source, entry['img_path'])
                specie_name = entry['specie_name']
                mask_path = os.path.join(self.source, entry['mask_path']) if entry['mask_path'] else None
                data_to_iterate.append([classname, specie_name, img_path, mask_path])

        return data_to_iterate
