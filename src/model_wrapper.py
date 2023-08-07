from os import path as os_path
import torch
import numpy as np
import cv2
import caffemodel2pytorch


class CaffeModel:
    def __init__(self, image_dir, device):
        self.image_dir = image_dir
        self.device = device

        self.scale = 224.0
        self.max_size = 1000.0
        # self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]], dtype=np.float32)

        self.classes_obj = None
        self.classes_attr = None
        self.model = None

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def prepare_input(self, image_id=None, image=None, bboxes=None):
        assert (image_id is not None and image is None) or (image_id is None and image is not None), \
            "Either image_id OR image must be not None."

        if image is None:
            image = cv2.imread(os_path.join(self.image_dir, image_id + '.jpg')).astype(dtype=np.float32)

        # Subtract the actual current pixel means - this makes more sense than using the same "means"
        # for every image, and it generates better results, especially for darker colors (including skin color)
        image = image - np.mean(image, axis=(0, 1,))

        img_shape = image.shape
        img_size_min = np.min(img_shape[0:2])
        img_size_max = np.max(img_shape[0:2])

        # Re-scale the image
        img_scale = self.scale / float(img_size_min)
        if np.round(img_scale * img_size_max) > self.max_size:
            img_scale = self.max_size / float(img_size_max)
        image = cv2.resize(src=image, dst=None, dsize=None, fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LINEAR)

        # Create a torch tensor and change dims to [batch_size, channels, height, width]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        image = image.permute((0, -1, 1, 2))

        im_info = torch.tensor([image.size(2), image.size(3), img_scale], dtype=torch.float32)

        if bboxes is not None:
            levels = np.zeros((len(bboxes), 1), dtype=np.float32)
            rois = np.asarray(bboxes, dtype=np.float32) * img_scale
            rois = torch.tensor(np.hstack((levels, rois)))
        else:
            rois = None

        return image, im_info, rois

    def load_model(self, prototxt_path, weights_path, caffe_proto_path):
        self.model = caffemodel2pytorch.Net(prototxt=prototxt_path, device=self.device, weights=weights_path,
                                            caffe_proto=caffe_proto_path)
        self.model.to(device=self.device)
        self.model.eval()

    def load_classes(self, class_files_dir):
        print('Preparing classes for objects, attributes and relations.')
        self.classes_obj = ['__background__']
        with open(os_path.join(class_files_dir, 'objects_vocab.txt')) as f:
            for obj in f.readlines():
                self.classes_obj.append(obj.lower().strip())

        self.classes_attr = ['___no_attr___']
        with open(os_path.join(class_files_dir, 'attributes_vocab.txt')) as f:
            for attr in f.readlines():
                self.classes_attr.append(attr.lower().strip())
