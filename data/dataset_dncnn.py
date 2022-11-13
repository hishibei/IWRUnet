
import os
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
from PIL import Image
import numpy
import torchvision.transforms.functional as tvF
import cv2


class DatasetDnCNN(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetDnCNN, self).__init__()
        print('Dataset: Denosing on AWGN with fixed sigma. Only dataroot_H is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma
		
		# 添加额外三个参数，透明度，覆盖率，水印地址
        self.transparency = opt['transparency'] if opt['transparency'] else random.randint(50, 80)
        self.coverage = opt['coverage'] if opt['coverage'] else 0
        self.water_path = opt['water_path']

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])

        # 低水平图片位置（如果需要）
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

    # 原版加水印函数
    # def add_watermark(self, image):

    #     # 读取随机水印
    #     water_path = '/home/shibei/WR2/water_imgs/'
    #     waters_list_path = [os.path.join(water_path, i) for i in os.listdir(water_path)]
    #     water_list_path = np.random.choice(waters_list_path, 1)[0]
    #     watermark_img = Image.open(water_list_path)

    #     # 水印旋转（1，21）
    #     watermark_img = watermark_img.rotate(-random.randint(1,21))
	
    #     # 水印透明度（60，90）
    #     TRANSPARENCY = random.randint(60, 90)
    #     paste_mask = watermark_img.split()[3].point(lambda i: i * TRANSPARENCY / 100.)
	
    #     # 添加水印
    #     image.paste(watermark_img, (0, 0), mask=paste_mask)

    #     return image

    # 更新版加水印函数
    def add_watermark(self, image):

        # 读取文件夹内随机水印
        water_path = self.water_path
        waters_list_path = [os.path.join(water_path, i) for i in os.listdir(water_path)]
        water_list_path = np.random.choice(waters_list_path, 1)[0]
        watermark_img = Image.open(water_list_path)

        # 设置覆盖率，默认（50，80）
        coverage = self.coverage

        # 读取水印的尺寸
        water_w , water_h = watermark_img.size

        # 读取图片的尺寸
        image_w , image_h = image.size

        # 设置纯水印图片
        img_water = np.zeros((image_h, image_w), np.uint8)
        img_water = Image.fromarray(img_water)

		# 根据覆盖度添加水印
        while True:

			# 打开水印
            watermark_img = Image.open(water_list_path)
			
            # 水印旋转(30°,-30°)
            angle = random.randint(-30,30)
            watermark_img = watermark_img.rotate(angle)

            #水印缩放(0.7，1.0)之间随机浮点数
            scale = random.uniform(0.7,1.0)
            watermark_img = watermark_img.resize((int(water_w * scale), int(water_h * scale)))

            # 水印透明度(默认（50，80）)
            TRANSPARENCY = self.transparency
            paste_mask = watermark_img.split()[3].point(lambda i: i * TRANSPARENCY / 100.)

            # 随机选择粘贴位置
            x = random.randint(int(-water_w * scale), image_w)
            y = random.randint(int(-water_h * scale), image_h)
			
            # 添加水印
            image.paste(watermark_img, (x, y), mask=paste_mask)
            img_water.paste(watermark_img, (x, y), mask=paste_mask)

            #返回检测（控制覆盖率）
            img_water = np.array(img_water)
            sum = (img_water > 0).sum()
            if sum > image_h * image_w * coverage / 100:
                break
            img_water = Image.fromarray(img_water)

        return image

	# 改变图片尺寸，送入tensor（默认512）
    def resize(self, img, H = 512 , W = 512):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """
        resized_img = tvF.resize(img, (H, W))
        return resized_img

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)    #读取的是数组类型 RGB H W C

        if self.opt['phase'] == 'train':

            L_path = H_path
            img_L = np.copy(img_H)

            img_L = Image.fromarray(np.uint8(img_L))  # 数组转为PIL类型  RGB W H C
            img_L= self.resize(img_L,self.patch_size,self.patch_size)
            img_L = self.add_watermark(img_L)
            img_L = numpy.array(img_L)  # PIL转换为数组类型  H W C

            img_H = Image.fromarray(np.uint8(img_H))  # 数组转为PIL类型  RGB W H C
            img_H= self.resize(img_H,self.patch_size,self.patch_size)
            img_H = numpy.array(img_H)  # PIL转换为数组类型  H W C

            img_H = util.uint2tensor3(img_H)
            img_L = util.uint2tensor3(img_L)

        else:

            # # 测试使用加好的水印
            # L_path = self.paths_L[index]
            # img_L = util.imread_uint(L_path, self.n_channels)    #读取的是数组类型 RGB H W C

            # img_L = Image.fromarray(np.uint8(img_L))  # 数组转为PIL类型  RGB W H C
            # img_L= self.resize(img_L,self.patch_size,self.patch_size)
            # img_L = numpy.array(img_L)  # PIL转换为数组类型  H W C

            # img_H = Image.fromarray(np.uint8(img_H))  # 数组转为PIL类型  RGB W H C
            # img_H= self.resize(img_H,self.patch_size,self.patch_size)
            # img_H = numpy.array(img_H)  # PIL转换为数组类型  H W C

            # img_H = util.uint2tensor3(img_H)
            # img_L = util.uint2tensor3(img_L)



            # 测试使用随机添加的水印
            L_path = H_path
            img_L = np.copy(img_H)

            img_L = Image.fromarray(np.uint8(img_L))  # 数组转为PIL类型  RGB W H C
            img_L= self.resize(img_L,self.patch_size,self.patch_size)
            img_L = self.add_watermark(img_L)
            img_L = numpy.array(img_L)  # PIL转换为数组类型  H W C

            img_H = Image.fromarray(np.uint8(img_H))  # 数组转为PIL类型  RGB W H C
            img_H= self.resize(img_H,self.patch_size,self.patch_size)
            img_H = numpy.array(img_H)  # PIL转换为数组类型  H W C

            img_H = util.uint2tensor3(img_H)
            img_L = util.uint2tensor3(img_L)


        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)
