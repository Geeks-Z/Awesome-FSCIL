import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None
    class_names = None
    class_ids = None


def _clean_folder_label(name):
    parts = name.split(".", 1)
    if len(parts) == 2 and parts[0].isdigit():
        name = parts[1]
    return name.replace("_", " ").replace("-", " ").strip()


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("/home/team/zhaohongwei/Dataset", train=True, download=False)
        test_dataset = datasets.cifar.CIFAR10("/home/team/zhaohongwei/Dataset", train=False, download=False)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
        self.class_names = list(train_dataset.classes)
        self.class_ids = list(train_dataset.classes)


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("/home/team/zhaohongwei/Dataset", train=True, download=False)
        test_dataset = datasets.cifar.CIFAR100("/home/team/zhaohongwei/Dataset", train=False, download=False)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
        self.class_names = list(train_dataset.classes)
        self.class_ids = list(train_dataset.classes)

def build_transform_coda_prompt(is_train, args):
    if is_train:
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
        return transform

    t = []
    if args["dataset"].startswith("imagenet"):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]

    return t

def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)

        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())

    # return transforms.Compose(t)
    return t

class iCIFAR224(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("/home/team/zhaohongwei/Dataset", train=True, download=False)
        test_dataset = datasets.cifar.CIFAR100("/home/team/zhaohongwei/Dataset", train=False, download=False)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
        self.class_names = list(train_dataset.classes)
        self.class_ids = list(train_dataset.classes)

class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        self.class_ids = list(train_dset.classes)
        self.class_names = [_clean_folder_label(name) for name in train_dset.classes]


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        self.class_ids = list(train_dset.classes)
        self.class_names = [_clean_folder_label(name) for name in train_dset.classes]


class iImageNetR(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/home/team/zhaohongwei/Dataset/imagenet-r/train/"
        test_dir = "/home/team/zhaohongwei/Dataset/imagenet-r/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        self.class_ids = list(train_dset.classes)
        self.class_names = [_clean_folder_label(name) for name in train_dset.classes]


class iImageNetA(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/home/team/zhaohongwei/Dataset/imagenet-a/train/"
        test_dir = "/home/team/zhaohongwei/Dataset/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        self.class_ids = list(train_dset.classes)
        self.class_names = [_clean_folder_label(name) for name in train_dset.classes]



class CUB(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/home/team/zhaohongwei/Dataset/cub/train/"
        test_dir = "/home/team/zhaohongwei/Dataset/cub/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        self.class_ids = list(train_dset.classes)
        self.class_names = [_clean_folder_label(name) for name in train_dset.classes]


class objectnet(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/home/team/zhaohongwei/Dataset/objectnet/train/"
        test_dir = "/home/team/zhaohongwei/Dataset/objectnet/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        self.class_ids = list(train_dset.classes)
        self.class_names = [_clean_folder_label(name) for name in train_dset.classes]


class omnibenchmark(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/home/team/zhaohongwei/Dataset/omnibenchmark/train/"
        test_dir = "/home/team/zhaohongwei/Dataset/omnibenchmark/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        self.class_ids = list(train_dset.classes)
        self.class_names = [_clean_folder_label(name) for name in train_dset.classes]



class vtab(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/home/team/zhaohongwei/Dataset/vtab/train/"
        test_dir = "/home/team/zhaohongwei/Dataset/vtab/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iMiniImageNet(iData):
    class_order = np.arange(100).tolist()

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        init_size = 256
        image_size = 224
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        from utils.autoaugment import AutoAugImageNetPolicy
        self.train_trsf = [
            transforms.Resize([init_size, init_size]),
            transforms.RandomResizedCrop(image_size),
            flip_and_color_jitter,
            AutoAugImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]

        self.test_trsf = [
            transforms.Resize([init_size, init_size]),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]

        self.common_trsf = []

    def download_data(self):
        import os.path as osp
        self.IMAGE_PATH = "/home/team/zhaohongwei/Dataset/miniimagenet/images"
        self.SPLIT_PATH = "/home/team/zhaohongwei/Dataset/miniimagenet/split"

        train_csv_path = osp.join(self.SPLIT_PATH, 'train.csv')
        text_csv_path = osp.join(self.SPLIT_PATH, 'test.csv')
        train_lines = [x.strip() for x in open(train_csv_path, 'r').readlines()][1:]
        test_lines = [x.strip() for x in open(text_csv_path, 'r').readlines()][1:]

        self.train_data, self.test_data = [], []
        self.train_targets, self.test_targets = [], []
        self.train_data2label, self.test_data2label = {}, {}

        lb = -1

        self.train_wnids, self.test_wnids = [], []

        for l in train_lines:
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.train_wnids:
                self.train_wnids.append(wnid)
                lb += 1
            self.train_data.append(path)
            self.train_targets.append(lb)
            # self.train_data2label[path] = lb
        # test
        lb = -1
        for l in test_lines:
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.test_wnids:
                self.test_wnids.append(wnid)
                lb += 1
            self.test_data.append(path)
            self.test_targets.append(lb)
            # self.test_data2label[path] = lb

        self.train_data = np.array(self.train_data)
        self.test_data = np.array(self.test_data)
        self.class_ids = list(self.train_wnids)
        self.class_names = list(self.train_wnids)
