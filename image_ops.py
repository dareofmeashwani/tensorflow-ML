from PIL import Image
import cv2
import numpy as np
class data_augmentor:
    def perspective_transform(self,images):
        def get_mask_coord(imshape):
            vertices = np.array([[(0.01 * imshape[1], 0.09 * imshape[0]),
                                  (0.13 * imshape[1], 0.32 * imshape[0]),
                                  (0.15 * imshape[1], 0.32 * imshape[0]),
                                  (0.89 * imshape[1], 0.5 * imshape[0])]], dtype=np.int32)
            return vertices

        def get_perspective_matrices(X_img):
            offset = 15
            img_size = (X_img.shape[1], X_img.shape[0])
            src = np.float32(get_mask_coord(X_img.shape))
            dst = np.float32([[offset, img_size[1]], [offset, 0], [img_size[0] - offset, 0],
                              [img_size[0] - offset, img_size[1]]])
            perspective_matrix = cv2.getPerspectiveTransform(src, dst)
            return perspective_matrix

        def transform(image):
            perspective_matrix = get_perspective_matrices(image)
            image = cv2.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]),
                                        flags=cv2.INTER_LINEAR)
            return image

        if len(images.shape) == 3:
            return transform(images)
        elif len(images.shape) == 4:
            img = []
            for i in range(len(images)):
                img.append(transform(images[i]))
            return np.array(img)

    def add_noise(self,images, type='blur'):
        def change(image):
            if (type == 'blur'):
                return cv2.blur(image, (5, 5))
            elif (type == 'gussian'):
                return cv2.GaussianBlur(image, (5, 5), 0)
            elif (type == 'median'):
                return cv2.medianBlur(image, 5)
            elif (type == 'bilateral'):
                return cv2.bilateralFilter(image, 9, 75, 75)

        if len(images.shape) == 3:
            return change(images)
        elif len(images.shape) == 4:
            img = []
            for i in range(len(images)):
                img.append(change(images[i]))
            return np.array(img)

    def add_salt_pepper_noise(self,images, salt_vs_pepper=0.2, amount=0.004):
        def change(X_imgs):
            X_imgs_copy = X_imgs.copy()
            row, col, _ = X_imgs_copy.shape
            num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
            num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))
            for X_img in X_imgs_copy:
                # Add Salt noise
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
                X_img[coords[0], coords[1]] = 1
                # Add Pepper noise
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
                X_img[coords[0], coords[1]] = 0
            return X_imgs_copy

        if len(images.shape) == 3:
            return change(images)
        elif len(images.shape) == 4:
            img = []
            for i in range(len(images)):
                img.append(change(images[i]))
            return np.array(img)

    def augment_brightness_camera_images(self,images, brightness):
        def change(image, bright):
            image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image1[:, :, 2] = image1[:, :, 2] * bright
            image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
            return image1

        if len(images.shape) == 3:
            return change(images, brightness)
        elif len(images.shape) == 4:
            img = []
            for i in range(len(images)):
                img.append(change(images[i], brightness))
            return np.array(img)

    def rotation(self,images, ang_range):
        def change(image):
            ang_rot = np.random.uniform(ang_range) - ang_range / 2
            rows, cols, ch = image.shape
            Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)
            image = cv2.warpAffine(image, Rot_M, (cols, rows))
            return image

        if len(images.shape) == 3:
            return change(images)
        elif len(images.shape) == 4:
            img = []
            for i in range(len(images)):
                img.append(change(images[i]))
            return np.array(img)

    def shear(self,images, shear_range):
        def change(image):
            rows, cols, ch = image.shape
            pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
            pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
            pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
            pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
            shear_M = cv2.getAffineTransform(pts1, pts2)
            image = cv2.warpAffine(image, shear_M, (cols, rows))
            return image

        if len(images.shape) == 3:
            return change(images)
        elif len(images.shape) == 4:
            img = []
            for i in range(len(images)):
                img.append(change(images[i]))
            return np.array(img)

    def translation(self,images, trans_range):
        def change(image):
            rows, cols, ch = image.shape
            tr_x = trans_range * np.random.uniform() - trans_range / 2
            tr_y = trans_range * np.random.uniform() - trans_range / 2
            Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
            image = cv2.warpAffine(image, Trans_M, (cols, rows))
            return image

        if len(images.shape) == 3:
            return change(images)
        elif len(images.shape) == 4:
            img = []
            for i in range(len(images)):
                img.append(change(images[i]))
            return np.array(img)

    def horizontal_flip(self,images):
        if len(images.shape) == 3:
            return cv2.flip(images, 0)
        elif len(images.shape) == 4:
            img = []
            for i in range(len(images)):
                img.append(cv2.flip(images[i], 0))
            return np.array(img)

    def vertical_flip(self,images):
        if len(images.shape) == 3:
            return cv2.flip(images, 1)
        elif len(images.shape) == 4:
            img = []
            for i in range(len(images)):
                img.append(cv2.flip(images[i], 1))
            return np.array(img)

    def transpose_flip(self,images):
        if len(images.shape) == 3:
            return cv2.flip(images, -1)
        elif len(images.shape) == 4:
            img = []
            for i in range(len(images)):
                img.append(cv2.flip(images[i], -1))
            return np.array(img)

    def change_contrast(self,images, level):
        factor = (259 * (level + 255)) / (255 * (259 - level))

        def contrast(c):
            return 128 + factor * (c - 128)

        if len(images.shape) == 3:
            return contrast(images)
        elif len(images.shape) == 4:
            img = []
            for i in range(len(images)):
                img.append(contrast(images[i]))
            return np.array(img)

    def saturation(self,images, ratio=0.5):
        import PIL.ImageEnhance as enhance
        if len(images.shape) == 3:
            images = Image.fromarray(images)
            converter = enhance.Color(images)
            return converter.enhance(ratio)
        elif len(images.shape) == 4:
            imgg = []
            for i in range(len(images)):
                img = Image.fromarray(images[i])
                converter = enhance.Color(img)
                imgg.append(np.array(converter.enhance(ratio)))
            return np.array(imgg)

    def convert_grayscale(self,images):
        def convert(image):
            image = Image.fromarray(image).convert('L')
            image = np.asarray(image, dtype="int32")
            return image

        if len(images.shape) == 3:
            return convert(images)
        elif len(images.shape) == 4:
            img = []
            for i in range(len(images)):
                img.append(convert(images[i]))
            return np.array(img)

    def convert_halftoning(self,images):
        def convert(image):
            im = Image.fromarray(image)
            cmyk = im.convert('CMYK').split()  # RGB contone RGB to CMYK contone
            c = cmyk[0].convert('1').convert('L')  # and then halftone ('1') each plane
            m = cmyk[1].convert('1').convert('L')  # ...and back to ('L') mode
            y = cmyk[2].convert('1').convert('L')
            k = cmyk[3].convert('1').convert('L')
            new_cmyk = Image.merge('CMYK', [c, m, y, k])  # put together all 4 planes
            return np.array(new_cmyk)

        if len(images.shape) == 3:
            return convert(images)
        elif len(images.shape) == 4:
            img = []
            for i in range(len(images)):
                img.append(convert(images[i]))
            return np.array(img)

    def rgb_to_hsv(self,rgb):
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select(
            [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    def hsv_to_rgb(self,hsv):
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def shift_hue(self,images, hout):
        if len(images.shape) == 3:
            hsv = self.rgb_to_hsv(images)
            hsv[..., 0] = hout
            return self.hsv_to_rgb(hsv)
        elif len(images.shape) == 4:
            img = []
            for i in range(len(images)):
                hsv = self.rgb_to_hsv(images[i])
                hsv[..., 0] = hout
                img.append(self.hsv_to_rgb(hsv))
            return np.array(img)

    def random_erasing(self,images, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        '''
        probability: The probability that the operation will be performed.
        sl: min erasing area
        sh: max erasing area
        r1: min aspect ratio
        mean: erasing value
        '''
        import random, math
        def erase(img):
            for attempt in range(50):
                area = img.shape[0] * img.shape[2]
                target_area = random.uniform(sl, sh) * area
                aspect_ratio = random.uniform(r1, 1 / r1)
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w <= img.shape[0] and h <= img.shape[1]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)
                    if img.shape[0] == 3:
                        img[x1:x1 + h, y1:y1 + w, 0] = mean[0]
                        img[x1:x1 + h, y1:y1 + w, 1] = mean[1]
                        img[x1:x1 + h, y1:y1 + w, 2] = mean[2]
                    else:
                        img[x1:x1 + h, y1:y1 + w, 0] = mean[0]
                    return img
            return img

        if len(images.shape) == 3:
            return erase(images)
        elif len(images.shape) == 4:
            img = []
            for i in range(len(images)):
                img.append(erase(images[i]))
            return np.array(img)
    def crop_from_centre(self,img,width,height):
        img=Image.fromarray(img)
        old_width, old_height = img.size
        left = (old_width - width)/2
        top = (old_height - height)/2
        right = (old_width + width)/2
        bottom = (old_height + height)/2
        img=img.crop((left, top, right, bottom))
        return np.array(img)

    def resize_image(self,img, width=80, height=80):
        img=Image.fromarray(img)
        img=img.resize((width,height), Image.ANTIALIAS)
        return np.array(img)

    def resize_image_with_aspect_ratio(self,img,size=128,padding=True,pad_option=0):
        height,width=img.shape[0],img.shape[1]
        if height>width:
            ratio=float(size)/height
            new_height=size
            new_width=int(ratio*width)
            img=self.resize_image(img,new_width,new_height)
            pad_size=size-new_width
            if padding:
                if pad_size%2==0:
                    img=np.pad(img, ((0,0), (pad_size//2, pad_size//2),(0,0)),'constant',constant_values=pad_option)
                else:
                    img = np.pad(img, ((0, 0), (pad_size//2, pad_size//2 +1), (0, 0)), 'constant', constant_values=pad_option)
        else:
            ratio=float(size)/width
            new_width=size
            new_height=int(ratio*height)
            img=self.resize_image(img,new_width,new_height)
            pad_size=size - new_height
            if padding:
                if pad_size%2==0:
                    img = np.pad(img, ((pad_size//2, pad_size//2), (0, 0), (0, 0)), 'constant', constant_values=pad_option)
                else:
                    img = np.pad(img, ((pad_size // 2, pad_size // 2 +1), (0, 0), (0, 0)), 'constant',
                                 constant_values=pad_option)
        return img

    def read_image(self,filename):
        from PIL import Image
        img = Image.open(filename)
        arr = np.array(img)
        return arr

    def pca_color_augmenataion(self,data):
        import numpy as np
        def data_aug(img, evecs_mat):
            mu = 0
            sigma = 0.1
            feature_vec = np.matrix(evecs_mat)
            se = np.zeros((3, 1))
            se[0][0] = np.random.normal(mu, sigma) * evals[0]
            se[1][0] = np.random.normal(mu, sigma) * evals[1]
            se[2][0] = np.random.normal(mu, sigma) * evals[2]
            se = np.matrix(se)
            val = feature_vec * se
            for i in xrange(img.shape[0]):
                for j in xrange(img.shape[1]):
                    for k in xrange(img.shape[2]):
                        img[i, j, k] = float(img[i, j, k]) + float(val[k])
            return img

        res = data.reshape([-1, 3])
        m = res.mean(axis=0)
        res = res - m
        R = np.cov(res, rowvar=False)
        from numpy import linalg as LA
        evals, evecs = LA.eigh(R)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        evals = evals[idx]
        evecs = evecs[:, :3]
        # evecs_mat = np.column_stack((evecs))
        m = np.dot(evecs.T, res.T).T
        img = []
        for i in range(len(data)):
            img.append(data_aug(data[i], m))
        return np.array(img)
