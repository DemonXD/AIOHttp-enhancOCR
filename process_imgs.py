from base64 import b64decode
import os
from typing import List

from mxnet.ndarray.gen_op import exp
from utils.tools.ocr_util import predict_text, predict
from utils.tools.pic_util import checkb64, b64Tndarray, img2b64


def RmSpeciContentImage(img_path: str) -> None:
    typs = [".png", ".jpg", ".jpeg", ".gif"]
    badFilesList = []
    NoneFileList = []
    for root, dirs, files in os.walk(img_path):
        # 检查当前目录中的损坏的图片文件
        for each in files:
            # for each in os.listdir('./'):
            if any(map(each.lower().endswith, typs)):
                try:
                    b64c = img2b64(os.path.join(root, each))
                    img = b64Tndarray(b64c)
                    res = predict(img)
                    if res == "无":
                        print(res)
                        NoneFileList.append(os.path.join(root, each))
                    else:
                        with open("labels.txt", "a+") as f:
                            f.write("{0}, {1}\n".format(os.path.join(root, each), res))
                            print(res)
                except Exception as e:
                    badFilesList.append(os.path.join(root, each))
    if len(badFilesList) > 0:
        for each in badFilesList:
            try:
                os.remove(each)
            except Exception as e:
                pass
    if len(NoneFileList) > 0:
        for each in NoneFileList:
            try:
                os.remove(each)
            except Exception:
                pass
        print("无 照片删除完毕")


if __name__ == "__main__":
    RmSpeciContentImage(r"D:\Tmp_Projects\AIOHttp-enhancOCR\imgs")