git pull --rebase origin master;
sudo docker stop ocr;
sudo docker build -t ocr:1.0 .;
# 加上--rm使镜像在退出时自动清除container
sudo docker run \
    --name ocr \
    -idt \
    --rm \
    -v /etc/localtime:/etc/localtime:ro \
    -v /var/www/logs:/var/www/ocr/logs \
    -p 8888:8888 ocr:1.0 \
    sh -c "python3 /var/www/ocr/app.py $1";
sudo docker logs ocr;
sudo docker rmi -f $(sudo docker images -f "dangling=true" -q)
sudo docker system prune --volumes --force