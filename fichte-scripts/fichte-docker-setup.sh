# build fichte docker images with this script 
# for local development use without Jenkins or another form of CICD


cd ~/fichte/flowers-emitter/src/main/docker/
docker build .

cd ~/fichte/flowers-registry/src/main/docker/
docker build .


