#!/bin/bash
# Define variables
CONTAINER_NAME="forum"
echo "resetting reddit"

docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME
docker run --rm -p 9999:80 ubuntu bash
sleep 1

docker run --name $CONTAINER_NAME -p 9999:80 --shm-size=2g --security-opt seccomp:unconfined -d postmill-populated-exposed-withimg
# wait ~15 secs for all services to start
sleep 15

echo "restarted reddit"
