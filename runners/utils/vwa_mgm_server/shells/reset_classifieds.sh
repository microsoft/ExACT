#!/bin/bash
# Define variables
echo "resetting classifieds"

docker stop classifieds_db classifieds
docker rm classifieds_db classifieds
echo "removed classifieds and classifieds_db"

cd <path_to_classifieds_dockerfile>
docker compose up --build -d
sleep 1
docker exec classifieds_db mysql -u root -ppassword osclass -e 'source docker-entrypoint-initdb.d/osclass_craigslist.sql'

# add restart policy
docker update --restart=always classifieds classifieds_db

# reset
export CLASSIFIEDS="<your_classifieds_domain>:9980"
curl -X POST ${CLASSIFIEDS}/index.php?page=reset -d "token=4b61655535e7ed388f0d40a93600254c"

echo "restarted classifieds"
