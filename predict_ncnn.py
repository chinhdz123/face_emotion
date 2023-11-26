import cv2
from ncnn.utils import print_topk
import numpy as np
import ncnn

class SqueezeNet:
    def __init__(self, target_size=66, num_threads=2, use_gpu=False):
        self.target_size = target_size
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        self.norm_vals = [1/255, 1/255, 1/255]
        self.mean_vals = []

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu

        # the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
        self.net.load_param("models/best_weights_model5.param")
        self.net.load_model("models/best_weights_model5.bin")

    def __del__(self):
        self.net = None

    def __call__(self, img):

        # mat_in = ncnn.Mat.from_pixels_resize(
        #     img,
        #     ncnn.Mat.PixelType.PIXEL_RGB,
        #     img.shape[1],
        #     img.shape[0],
        #     self.target_size,
        #     self.target_size,
        # )
        img = cv2.resize(img, (66,66))
        mat_in = ncnn.Mat.from_pixels(
            img,
            ncnn.Mat.PixelType.PIXEL_RGB,
            img.shape[1],
            img.shape[0],
        )
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)

        ex = self.net.create_extractor()
        ex.set_num_threads(self.num_threads)

        ex.input("input_5_blob", mat_in)

        ret, mat_out = ex.extract("dense_20_Softmax_blob")

        # printf("%d %d %d\n", mat_out.w, mat_out.h, mat_out.c)

        out = np.array(mat_out)
        return out
    




