import io
import base64
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


def check_pic_valid(imgbuf):
    """[summary]

    Args:
        img ([BytesIO]): [文件流]
    """
    try:
        Image.open(io.BytesIO(imgbuf)).verify()
    except:
        return False
    return True


def b64Tndarray(b64str):
    img = base64.b64decode(b64str)
    nparr = np.fromstring(img, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np


def ndarray2b64(ndarr):
    retval, buffer = cv2.imencode('.jpg', ndarr)
    pic_str = base64.b64encode(buffer)
    pic_str = pic_str.decode()


def img2b64(path):
    with open(path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        req_file = encoded_image.decode("utf8")
    return req_file


def Bytes2ndarray(byts):
    """[summary]

    Args:
        byts ([type]): [description]

    Returns:
        [type]: [description]
    
    Desc.:
        image file bytes to opencv img format(BGR)
    """
    return np.array(Image.open(BytesIO(byts)).convert("BGR"))


def checkb64(imgstr):
    try:
        base64.b64decode(imgstr)
    except Exception:
        return False, "该字符串是无效的base64图片编码"
    else:
        return True, None