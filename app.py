import os
import argparse
from aiohttp import web
from global_ import PublicManager
from router import init_urls
from config.config_base import Config
from config.initconf import (
    init_log,
    init_dbpool,
    init_redis,
    init_mqtt
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(BASE_DIR, "config.yaml")


async def before_app_run(app):
    """[summary]

    Args:
        app ([type]): [项目实例]
    Desc.: 
        这里可以加载一些公共资源以便项目使用：
        app["pubdic"] = {...}
    """
    # init db config
    # init_dbpool(PublicManager, "pg", typ="postgresql")

    # await init_dbpool(PublicManager, "mysql", typ="mysql")
    # await PublicManager.dbpool["mysql"].connect()
    # await init_dbpool(PublicManager, "mongo", typ="mongo")
    # await PublicManager.dbpool["mongo"].connect()
    # init mqtt config
    # init_mqtt(PublicManager)
    # init redis
    # await init_redis(PublicManager)
    # await PublicManager.redisdb.connect()

    
async def after_app_stop(app):
    """[summary]

    Args:
        app ([type]): [description]
    Desc.:
        这里释放公共资源等
    """
    # await PublicManager.dbpool['postgresql'].close()
    # await PublicManager.dbpool['mysql'].close()
    # await PublicManager.redisdb.close()

# def initArges():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "env",
#         help="env => [test|development|production]",
#         nargs="?",
#         default="development"
#     )
#     args = parser.parse_args()
#     return args.env


# args = initArges()
# register conf
PublicManager.conf = Config(config_file, "development")
# init log config, 优先初始化log
init_log(PublicManager)


app = web.Application()
init_urls(app)

app.on_startup.append(before_app_run)
app.on_shutdown.append(after_app_stop)



if __name__=="__main__":
    try:
        web.run_app(app, host="0.0.0.0", port=8888)
    except KeyboardInterrupt:
        pass
