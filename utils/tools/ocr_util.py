# import numpy as np
import os
import numpy as np
from cnocr import CnOcr
from io import BytesIO
from PIL import Image
from .np_util import scale_resize, dnn_scale


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ocr = CnOcr(root=os.path.join(BASE_DIR, "cnocr_model"))


def predict_text(img: bytes):
    # imgs = cv2.imdecode(np.frombuffer(img, np.uint8), -1)
    imgs = np.array(Image.open(BytesIO(img)).convert("RGB"))


    # 利用opencv 将图片放大1.1倍
    # imgs = scale_resize(imgs, 150)
    
    # Image Enhancement
    # dnn_scale
    imgs = dnn_scale(imgs, 1)
    result = ocr.ocr(imgs)
    result_map = {
        0: None,
        1: "".join(result[0]),
    }
    result_ = result_map.get(len(result), "".join(["".join(x) for x in result])).replace(" ", "")

    return result_ if result_ is not None else "未识别到内容"


def predict(img):
    # imgs = scale_resize(img, 110)
    imgs = dnn_scale(img, 1)
    result = ocr.ocr(imgs)
    result_map = {
        0: None,
        1: "".join(result[0]),
    }
    result_ = result_map.get(len(result), "".join(["".join(x) for x in result])).replace(" ", "")
    return result_ if result_ is not None else "未识别到内容"
