# build and push
docker build -t xxx/k8s-mxnet-gpu:latest -f k8s-mxnet-gpu.Dockerfile .
docker push xxx/k8s-mxnet-gpu:latest