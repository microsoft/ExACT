#!/bin/bash
# Define variables
CONTAINER_NAME="shopping_admin"
echo "resetting CMS"

docker stop ${CONTAINER_NAME}
docker remove ${CONTAINER_NAME}
sleep 3
docker run --rm -p 7780:80 ubuntu bash

docker run --name ${CONTAINER_NAME} -p 7780:80 --shm-size=2g --security-opt seccomp:unconfined -d shopping_admin_final_0719
echo "wait ~1 min to wait all services to start"
sleep 60

docker exec ${CONTAINER_NAME} /var/www/magento2/bin/magento setup:store-config:set --base-url="<your_e_commerce_cms_domain>:7780" # no trailing slash
docker exec ${CONTAINER_NAME} mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="<your_e_commerce_cms_domain>:7780/" WHERE path = "web/secure/base_url";'
docker exec ${CONTAINER_NAME} /var/www/magento2/bin/magento cache:flush
sleep 1

echo "restarted CMS"