import cv2
import random
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from .const import SETS, data_dir

# 生成椒盐噪声
def img_salt_pepper_noise(src,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        if random.randint(0,1)==0:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg

# 随机生成不定长图片集
def gen_text(cnt):
    # 设置文字字体和大小
    font_path = '/data/work/tensorflow/fonts/arial.ttf'
    font_size = 30
    font=ImageFont.truetype(font_path,font_size)

    for i in range(cnt):
            # 随机生成1到10位的不定长数字
            rnd = random.randint(1, 10)
            text = ''
            for j in range(rnd):
                text = text + SETS[random.randint(0, len(SETS) - 1)]

    # 生成图片并绘上文字
            img=Image.new("RGB",(256,32))
            draw=ImageDraw.Draw(img)
            draw.text((1,1),text,font=font,fill='white')
            img=np.array(img)

    # 随机叠加椒盐噪声并保存图像
            img = img_salt_pepper_noise(img, float(random.randint(1,10)/100.0))
            cv2.imwrite(data_dir + text + '_' + str(i+1) + '.jpg',img)


if __name__ == "__main__":
    gen_text(10000)