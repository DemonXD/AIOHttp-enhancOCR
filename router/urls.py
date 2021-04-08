import aiohttp_cors
from handlers.http_handler import handle
from handlers.ocr_handler import ocr_handle, ocr_handle_b64
from handlers.ws_handler import wshandle


def init_urls(app):
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
    })
    cors.add(app.router.add_get("/ping", handle))
    cors.add(app.router.add_get("/echo", wshandle))
    cors.add(app.router.add_post("/ocr_file", ocr_handle))
    cors.add(app.router.add_post("/ocr_b64", ocr_handle_b64))
    return app