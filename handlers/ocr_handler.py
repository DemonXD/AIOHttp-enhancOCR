from aiohttp.web import FileField
from aiohttp.web import json_response
from global_ import PublicManager
from utils.tools.ocr_util import predict_text, predict
from utils.tools.pic_util import checkb64, b64Tndarray


async def ocr_handle(request):
    # 获取post body 数据
    # content = await request.json()
    try:
        data = await request.post()
        file_ = data.get("image_file", None)
        if file_ is not None and isinstance(file_, FileField):
            res = predict_text(file_.file.read())
        else:
            raise Exception("文件不能为空")
    except Exception as e:
        PublicManager.logger.error(str(e))
        return json_response({"code": "4001", "msg": "failure", "data":str(e)})
    else:
        return json_response({"code": "2001", "msg": "success", "data": res})


async def ocr_handle_b64(request):
    try:
        data = await request.json()
        img = data.get("image_file", None)
        assert img is not None, "不存在附件"
        is_, msg = checkb64(img)
        if is_:
            img = img.encode("utf8")
            img = b64Tndarray(img)
            res = predict(img)
        else:
            return json_response({"code": "4001", "msg":"failure", "data": msg})
    except Exception as e:
        PublicManager.logger.error(str(e))
        return json_response({"code": "4001", "msg":"failure", "data":str(e)})

    else:
        return json_response({"code": "2001", "msg": "success", "data": res})