from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import math
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFilter
from synthesis import transform_image
import cv2
import random


mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

class Resize(object):

    def __init__(self, size=None, scale_factor=None):
        self.size = size
        self.scale_factor = scale_factor

    def __call__(self, data):
        image, label = data['image'], data['label']

        if self.scale_factor is not None:
            new_width = int(image.width * self.scale_factor)
            new_height = int(image.height * self.scale_factor)
            new_width, new_height = resize_to_multiple_of_32(new_width, new_height)
            image = F.resize(image, (new_height, new_width))
            label = F.resize(label, (new_height, new_width), interpolation=InterpolationMode.BICUBIC)
        elif self.size is not None:
            image = F.resize(image, self.size)
            # print(image.shape())
            label = F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC)
        else:
            raise ValueError("Either size or scale_factor must be provided")

        return {'image': image, 'label': label}
def resize_to_multiple_of_32(width, height):
    new_width = math.ceil(width / 32) * 32
    new_height = math.ceil(height / 32) * 32
    return new_width, new_height


def random_brightness_region(img):
    shape = 'polygon'  
    
    width, height = img.size
    min_size = min(width, height) // 10
    max_size = min(width, height) // 3
    region_size_w = np.random.randint(min_size, max_size)
    region_size_h = np.random.randint(min_size, max_size)

    x = np.random.randint(0, width - region_size_w)
    y = np.random.randint(0, height - region_size_h)

    brightness_factor = np.random.uniform(0.2, 2)

    mask = Image.new('L', (region_size_w, region_size_h), 0)
    draw = ImageDraw.Draw(mask)

    if shape == 'polygon':
        num_points = 20  
        center_x, center_y = region_size_w // 2, region_size_h // 2
        radius_base = min(region_size_w, region_size_h) // 2 * 0.8 
        points = []

        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            radius = radius_base * np.random.uniform(0.5, 1.0)
            px = center_x + int(radius * np.cos(angle))
            py = center_y + int(radius * np.sin(angle))
            points.append((px, py))

        draw.polygon(points, fill=255)
    elif shape == 'rect':
        draw.rectangle((0, 0, region_size_w, region_size_h), fill=255)
    else:
        draw.ellipse((0, 0, region_size_w, region_size_h), fill=255)

    mask = mask.filter(ImageFilter.GaussianBlur(radius=region_size_w // 5))

    region = img.crop((x, y, x + region_size_w, y + region_size_h))
    enhancer = ImageEnhance.Brightness(region)
    region = enhancer.enhance(brightness_factor)

    img.paste(region, (x, y), mask)

    return img

def get_trainData_transforms(ratio):
    data_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.resize((
            ((int(img.width*ratio) + 7) // 14) * 14,
            ((int(img.height*ratio) + 7) // 14) * 14
        ))),
        transforms.Lambda(random_brightness_region),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return data_transform


def get_data_transforms(ratio):
    data_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.resize((
            ((int(img.width*ratio) + 7) // 14) * 14,
            ((int(img.height*ratio) + 7) // 14) * 14
        ))),
        # transforms.Lambda(random_brightness_region),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    gt_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.resize((
            ((int(img.width*ratio) + 7) // 14) * 14,
            ((int(img.height*ratio) + 7) // 14) * 14
        ))),
        transforms.ToTensor()
    ])
    return data_transform, gt_transform

def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.

def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.Resampling.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)

def create_walnuts(img, severity):

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    height, width, _ = img_cv.shape
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, dark_regions = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    dark_regions1 = cv2.bitwise_not(dark_regions)
    kernel = np.ones((15,15), np.uint8)
    dark_regions1 = cv2.dilate(dark_regions1, kernel, iterations=1)
    dark_regions = cv2.bitwise_not(dark_regions1)
    contours, _ = cv2.findContours(dark_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10000]

    if not contours:
        return img

    selected_contour = random.choice(contours)
    x, y, w, h = cv2.boundingRect(selected_contour)
    mask = np.zeros((height, width), dtype=np.uint8)

    def gaussian_weight(radius, max_radius, sigma):
        return math.exp(-(radius**2) / (2 * (max_radius * sigma)**2))

    center_x = x + w // 2
    center_y = y + h // 2
    scale_factor = 0.5  
    max_radius = int(min(w, h) // 2 * scale_factor)
    for radius in range(max_radius, 0, -1):
        weight = gaussian_weight(radius, max_radius, 0.35)
        intensity = int(255 * weight * 0.3)
        cv2.circle(mask, (center_x, center_y), radius, int(intensity), -1)

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=10)
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    dark_regions_3d = cv2.cvtColor(dark_regions, cv2.COLOR_GRAY2BGR)
    masked_shadow = cv2.bitwise_and(mask_3d, dark_regions_3d)
    result = cv2.addWeighted(img_cv, 1, masked_shadow, -0.5, 0)
    result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    return result_pil

def create_can(img, level):

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    highlight = np.zeros_like(img_cv)

    top_left_x = random.randint(300, 1800)
    top_left_y = random.randint(300, 800)
    bottom_right_x = top_left_x + 300
    bottom_right_y = top_left_y + 30

    cv2.rectangle(highlight, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 255, 255), -1)

    highlight = cv2.GaussianBlur(highlight, (0, 0), sigmaX=10, sigmaY=10)
    illuminated_image = cv2.addWeighted(img_cv, 1, highlight, 0.5, 0)
    img_pil = Image.fromarray(cv2.cvtColor(illuminated_image, cv2.COLOR_BGR2RGB))

    return img_pil

def create_fabric(pil_img, level):
    
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    height, width, _ = img.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    rect_width = int(width * 0.1)
    rect_x_start = np.random.randint(0, width - rect_width)
    rect_x_end = rect_x_start + rect_width
    rect_y_start = 0
    rect_y_end = height

    m = (rect_y_end - rect_y_start) / (rect_x_end - rect_x_start) if (rect_x_end - rect_x_start) > 0 else 0
    b = rect_y_start - m * rect_x_start

    for y in range(height):
        for x in range(width):
            distance = abs(y - (m * x + b))
            shadow_intensity = max(0, 0.5 - distance / 2500) 
            mask[y, x] = int(shadow_intensity * 200)

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=5, sigmaY=5)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    result_img = cv2.addWeighted(img, 1, mask, -0.5, 0)
    result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))

    return result_pil

def augmvtec(image, preprocess, severity=3, width=3, depth=-1, alpha=1.):
    aug_list = [brightness, contrast]

    severity = np.random.randint(0, severity)

    ws = np.float32(np.random.dirichlet([1] * width))
    m = np.float32(np.random.beta(alpha, alpha))
    preprocess_img = preprocess(image)
    mix = torch.zeros_like(preprocess_img)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, severity)
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess_img + m * mix
    return mixed

def augmvtecF(image,  preprocess, severity=3, width=1, depth=-1, alpha=1.):
    aug_list = [brightness, contrast, create_fabric]

    severity = np.random.randint(0, severity)

    ws = np.float32(np.random.dirichlet([1] * width))
    m = np.float32(np.random.beta(alpha, alpha))
    preprocess_img = preprocess(image)
    mix = torch.zeros_like(preprocess_img)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, severity)
        mix += ws[i]

    mixed = (1 - m) * preprocess_img + m * mix
    return mixed

def augmvtecC(image, preprocess, severity=3, width=1, depth=-1, alpha=1.):
    aug_list = [brightness, contrast, create_can]

    severity = np.random.randint(0, severity)

    ws = np.float32(np.random.dirichlet([1] * width))
    m = np.float32(np.random.beta(alpha, alpha))
    preprocess_img = preprocess(image)
    mix = torch.zeros_like(preprocess_img)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, severity)
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess_img + m * mix
    return mixed

def augmvtecW(image, preprocess, severity=3, width=1, depth=-1, alpha=1.):
    aug_list = [brightness, contrast, create_walnuts]

    severity = np.random.randint(0, severity)

    ws = np.float32(np.random.dirichlet([1] * width))
    m = np.float32(np.random.beta(alpha, alpha))
    preprocess_img = preprocess(image)
    mix = torch.zeros_like(preprocess_img)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, severity)
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess_img + m * mix
    return mixed

        
class AugMixDatasetMVTec(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, _class_):
    self.dataset = dataset
    self._class_ = _class_
    self.preprocess = preprocess
    self.gray_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train),
        transforms.Grayscale(3)
    ])

  def __getitem__(self, i):
    x, _ = self.dataset[i]
    if self._class_ == 'fabric':
        if random.random() > 0.2:
            return self.preprocess(x)
        else:
            return augmvtecF(x, self.preprocess)
    elif self._class_ == 'walnuts':
        if random.random() > 0.2:
            return self.preprocess(x)
        else:
            return augmvtecW(x, self.preprocess)
    elif self._class_ == 'can':
        if random.random() > 0.2:
            return self.preprocess(x)
        else:
            return augmvtecC(x, self.preprocess)
    else:
        if random.random() > 0.2:
            return self.preprocess(x)
        else:
            return augmvtec(x, self.preprocess)

  def __len__(self):
    return len(self.dataset)

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            # self.img_path = os.path.join(root, 'bad')
            # self.gt_path = os.path.join(root, 'ground_truth/bad')
            # print(self.img_path, self.gt_path)
            self.img_path = root
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = ['bad', 'good']
        # print(defect_types)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
                # tot_types.extend([os.path.splitext(os.path.basename(img_path))[0] for img_path in img_paths])
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.img_path, 'ground_truth/bad') + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))
                # tot_types.extend([os.path.splitext(os.path.basename(img_path))[0] for img_path in img_paths])

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-1]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_type

class MVTecTestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):


        self.img_path = root
        self.transform = transform
        # load dataset
        self.img_paths, self.img_name, self.img_type = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_name = []
        tot_types = []

        defect_types = ['test_private', 'test_private_mixed']
        # print(defect_types)

        for defect_type in defect_types:
            img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
            img_tot_paths.extend(img_paths)
            tot_name.extend([os.path.basename(img_path) for img_path in img_paths])
            tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(tot_name), "Something wrong with test and ground truth pair!"

        return img_tot_paths, tot_name, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, img_name, img_type = self.img_paths[idx], self.img_name[idx], self.img_type[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, img_name, img_type
    
def load_data(dataset_name='mnist',normal_class=0,batch_size='16'):

    if dataset_name == 'cifar10':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            #transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        os.makedirs("./Dataset/CIFAR10/train", exist_ok=True)
        dataset = CIFAR10('./Dataset/CIFAR10/train', train=True, download=True, transform=img_transform)
        print("Cifar10 DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/CIFAR10/test", exist_ok=True)
        test_set = CIFAR10("./Dataset/CIFAR10/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif dataset_name == 'mnist':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        os.makedirs("./Dataset/MNIST/train", exist_ok=True)
        dataset = MNIST('./Dataset/MNIST/train', train=True, download=True, transform=img_transform)
        print("MNIST DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/MNIST/test", exist_ok=True)
        test_set = MNIST("./Dataset/MNIST/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif dataset_name == 'fashionmnist':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        os.makedirs("./Dataset/FashionMNIST/train", exist_ok=True)
        dataset = FashionMNIST('./Dataset/FashionMNIST/train', train=True, download=True, transform=img_transform)
        print("FashionMNIST DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/FashionMNIST/test", exist_ok=True)
        test_set = FashionMNIST("./Dataset/FashionMNIST/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)


    elif dataset_name == 'retina':
        data_path = 'Dataset/OCT2017/train'

        orig_transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor()
        ])

        dataset = ImageFolder(root=data_path, transform=orig_transform)

        test_data_path = 'Dataset/OCT2017/test'
        test_set = ImageFolder(root=test_data_path, transform=orig_transform)

    else:
        raise Exception(
            "You enter {} as dataset, which is not a valid dataset for this repository!".format(dataset_name))

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
    )

    return train_dataloader, test_dataloader



class SynAugMixDatasetMVTec(torch.utils.data.Dataset):

    def __init__(self, dataset, preprocess, dtd_root):
        self.dataset = dataset
        self.preprocess = preprocess
        self.anomaly_source_paths = sorted(glob.glob(f"{dtd_root}/*/*.jpg"))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x_ori, _ = self.dataset[idx]
        x = np.array(x_ori)   
        x, mask, _ = transform_image(x, None, self.anomaly_source_paths, idx)  # 不需要mask

        x = Image.fromarray(np.uint8(x))

        return self.preprocess(x_ori), augmvtec(x, self.preprocess), mask


