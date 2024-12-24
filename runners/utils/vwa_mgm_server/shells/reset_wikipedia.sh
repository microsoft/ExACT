#!/bin/bash
# Define variables
CONTAINER_NAME="wikipedia"
echo "resetting wikipedia"

docker stop ${CONTAINER_NAME}
docker remove ${CONTAINER_NAME}
sleep 3
docker run --rm -p 8888:80 ubuntu bash

docker run -d --name=wikipedia \
--volume=<path_to_wikipedia_en_all_maxi_2022-05.zim>:/data \
--shm-size=2g \
--security-opt seccomp:unconfined \
-p 8888:80 ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2022-05.zim

# add restart policy
docker update --restart=always wikipedia

echo "restarted wikipedia"
