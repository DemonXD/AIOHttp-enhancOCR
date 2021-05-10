# import numpy as np
import os
import numpy as np
from cnocr import CnOcr
from io import BytesIO
from PIL import Image
from .np_util import scale_resize, dnn_scale
from .pic_util import check_pic_valid


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ocr = CnOcr(
    root=os.path.join(BASE_DIR, "cnocr_model"),
    model_name="densenet-lite-gru")


def predict_text(img: bytes):
    checked = check_pic_valid(img)
    if checked:
        imgs = np.array(Image.open(BytesIO(img)).convert("BGR"))

        # Image Enhancement
        # dnn_scale
        imgs = dnn_scale(imgs, 0.5)
        result = ocr.ocr(imgs)
        result_map = {
            0: None,
            1: "".join(result[0]),
        }
        result_ = result_map.get(len(result), "".join(["".join(x) for x in result])).replace(" ", "")

        return result_ if result_ is not None else "未识别到内容"
    return "无效的文件"

def predict(img):
    imgs = dnn_scale(img, 0.5)
    result = ocr.ocr(imgs)
    result_map = {
        0: None,
        1: "".join(result[0]),
    }
    result_ = result_map.get(len(result), "".join(["".join(x) for x in result])).replace(" ", "")
    return result_ if result_ is not None else "未识别到内容"
