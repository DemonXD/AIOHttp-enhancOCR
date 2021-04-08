import os
import cv2
import numpy
import numpy as np
from cv2 import dnn_superres


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel(os.path.join(BASE_DIR, "cv_model", "FSRCNN_x4.pb"))
sr.setModel("fsrcnn", 4)

def dnn_scale(img, perc):
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


if __name__ == "__main__":
    import platform
    if platform.platform().startswith("Windows"):
        image = cv2.imread(r"C:\Users\ci24924\Desktop\22.jpg")
    else:
        image = cv2.imread(r'/mnt/c/Users/ci24924/Desktop/22.jpg')
    dst = numpy.ones((image.shape[0],image.shape[1]),dtype = numpy.int8)

    scale_percent = 150       # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    # resized = cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)
    # im = scale_resize(image, 150)
    im = dnn_scale(image, 1.2)


    # im = Image.fromarray(im)

    cv_show("test", im)
    # im.save(r"D:\Career_Projects\career_ocr\resources\2021031211111111.png")