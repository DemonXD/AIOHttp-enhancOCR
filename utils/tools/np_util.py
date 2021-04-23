import os
import cv2
import numpy
import numpy as np
from cv2 import dnn_superres


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel(os.path.join(BASE_DIR, "cv_model", "FSRCNN_x4.pb"))
sr.setModel("fsrcnn", 4)

def rnoise(img):
    """[summary]
    Args:
        img ([image]): [description]
    Desc.:
        去噪
        fastNlMeansDenoisingColored参数
        - h：决定滤波器强度的参数。较高的h值可以更好地消除噪点，但同时也可以消除图像细节。（可以设为10）
        - hForColorComponents：与h相同，但仅用于彩色图像。（通常与h相同）
        - templateWindowSize：应为奇数。（建议设为7）
        - searchWindowSize：应为奇数。（建议设为21）
    """
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def img_threshold(img):
    """
        图像二值化
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_TOZERO)
    return binary


def dnn_scale(img, perc):
    img = rnoise(img)
    img = img_threshold(img)
    img_ = sr.upsample(img)
    res = cv2.resize(img_, None, fx=perc, fy=perc, interpolation=cv2.INTER_LINEAR)
    return res


def scale_resize(img, percent):
    scale_percent = percent       # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
    return resized


def nearest_resize(img, src_size):
    h, w, c = img.shape
    src = np.zeros((src_size[0], src_size[1], 3), dtype=np.uint8)
    if h == src_size[0] and w == src_size[1]:
        return img
    for i in range(src_size[0]):
        for j in range(src_size[1]):
            # round()四舍五入的函数
            src_x = round(i * (h / src_size[0]))
            src_y = round(j * (w / src_size[1]))
            src[i, j] = img[src_x, src_y]
    return src


def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def bc_interpolate(img, ax=1., ay=1.):
    H, W, C = img.shape
    aH = int(ay * H)
    aW = int(ax * W)

    # get position of resized image
    y = np.arange(aH).repeat(aW).reshape(aW, -1)
    x = np.tile(np.arange(aW), (aH, 1))

    # get position of original position
    y = y / ay
    x = x / ax

    ix = np.floor(x).astype(np.int)
    iy = np.floor(y).astype(np.int)

    ix = np.minimum(ix, W - 2)
    iy = np.minimum(iy, H - 2)

    # get distance
    dx2 = x - ix
    dy2 = y - iy
    dx1 = dx2 + 1
    dy1 = dy2 + 1
    dx3 = 1 - dx2
    dy3 = 1 - dy2
    dx4 = 1 + dx3
    dy4 = 1 + dy3

    dxs = [dx1, dx2, dx3, dx4]
    dys = [dy1, dy2, dy3, dy4]

    # bi-cubic weight
    def weight(t):
        a = -1.
        at = np.abs(t)
        w = np.zeros_like(t)
        ind = np.where(at <= 1)
        w[ind] = ((a + 2) * np.power(at, 3) - (a + 3) * np.power(at, 2) + 1)[ind]
        ind = np.where((at > 1) & (at <= 2))
        w[ind] = (a * np.power(at, 3) - 5 * a * np.power(at, 2) + 8 * a * at - 4 * a)[ind]
        return w

    w_sum = np.zeros((aH, aW, C), dtype=np.float32)
    out = np.zeros((aH, aW, C), dtype=np.float32)

    # interpolate
    for j in range(-1, 3):
        for i in range(-1, 3):
            ind_x = np.minimum(np.maximum(ix + i, 0), W - 1)
            ind_y = np.minimum(np.maximum(iy + j, 0), H - 1)

            wx = weight(dxs[i + 1])
            wy = weight(dys[j + 1])
            wx = np.repeat(np.expand_dims(wx, axis=-1), 3, axis=-1)
            wy = np.repeat(np.expand_dims(wy, axis=-1), 3, axis=-1)

            w_sum += wx * wy
            out += wx * wy * img[ind_y, ind_x]

    out /= w_sum
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out




if __name__ == "__main__":
    import sys
    import platform
    if platform.platform().startswith("Windows"):
        image = cv2.imread(r"{0}".format(sys.argv[1]))
    else:
        image = cv2.imread(r"{0}".format(sys.argv[1]))

    print(sys.argv[1])

    dst = numpy.ones((image.shape[0],image.shape[1]),dtype = numpy.int8)

    scale_percent = 150       # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    # resized = cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)
    # im = scale_resize(image, 150)
    im = dnn_scale(image, 1.1)


    # im = Image.fromarray(im)

    cv_show("test", im)
    # im.save(r"D:\Career_Projects\career_ocr\resources\2021031211111111.png")