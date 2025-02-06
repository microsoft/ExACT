docker rm -v $(docker stop $(docker ps -a -q --filter ancestor=happysixd/osworld-docker --format="{{.ID}}"))

rm -rf .lock/*