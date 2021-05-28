import os
import re
import fire
from base64 import b64decode
from typing import List
from tqdm import tqdm, tqdm_notebook

from threading import Thread
from mxnet.ndarray.gen_op import exp, random_generalized_negative_binomial
from utils.tools.ocr_util import predict_text, predict
from utils.tools.pic_util import checkb64, b64Tndarray, img2b64


def RmSpeciContentImage(img_path: str) -> None:
    typs = [".png", ".jpg", ".jpeg", ".gif"]
    badFilesList = []
    NoneFileList = []
    count = 1
    for root, dirs, files in os.walk(img_path):
        # 检查当前目录中的损坏的图片文件
        for each in tqdm(files, "Realizing ...", total=len(files)):
            # for each in os.listdir('./'):
            if any(map(each.lower().endswith, typs)):
                try:
                    b64c = img2b64(os.path.join(root, each))
                    img = b64Tndarray(b64c)
                    res = predict(img)

                    # new_name = ".".join([each.split(".")[0]+"_"+res, each.split(".")[-1]])
                    new_name = ".".join([str(count)+"_"+res, each.split(".")[-1]])
                    if res == "无":
                        NoneFileList.append(os.path.join(root, each))
                    else:
                        os.rename(os.path.join(root, each), os.path.join(root, new_name))
                        with open("labels.txt", "a+") as f:
                            f.write("{0}, {1}\n".format(os.path.join(root, new_name), res))
                    count += 1
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


def reNameImg(img_path: str) -> None:
    typs = [".png", ".jpg", ".jpeg", ".gif"]
    count = 0
    replace_str_pattern = re.compile(r"corm|corn|con", re.S)
    replace_str_pattern_hotmail = re.compile(r"hotmaill|hotmall|hotrmail", re.S)
    replace_str_pattern1_foxmail = re.compile(r"foxmall|foxmaill|foxrmail", re.S)
    replace_str_pattern1_gmail = re.compile(r"gmall|gmaill", re.S)
    replace_str_pattern1_aliyun = re.compile(r"allyun|alliyun", re.S)

    
    for root, dirs, files in os.walk(img_path):
        # 检查当前目录中的损坏的图片文件
        for each in tqdm(files):
            old_name = os.path.join(root, each)
            new_name = re.sub(replace_str_pattern, "com", old_name)
            new_name = re.sub(replace_str_pattern_hotmail, "hotmail", new_name)
            new_name = re.sub(replace_str_pattern1_foxmail, "foxmail", new_name)
            new_name = re.sub(replace_str_pattern1_gmail, "gmail", new_name)
            new_name = re.sub(replace_str_pattern1_aliyun, "aliyun", new_name)
            new_name.replace("I", "l")
            if old_name == new_name:
                pass
            else:
                os.rename(old_name, new_name)
            # # for each in os.listdir('./'):
            # if any(map(each.lower().endswith, typs)):
            #     count+=1
            #     file_name =  ".".join([str(count), each.split(".")[-1]])
            #     try:
            #         os.rename(os.path.join(root, each), os.path.join(root, file_name))
            #     except Exception:
            #         pass
    print(f"转换完毕")

if __name__ == "__main__":
    # RmSpeciContentImage(r"D:\Tmp_Projects\AIOHttp-enhancOCR\imgs")
    # reNameImg(r"D:\Tmp_Projects\AIOHttp-enhancOCR\imgs")
    fire.Fire({
        "Label": RmSpeciContentImage,
        "Rename": reNameImg
    })