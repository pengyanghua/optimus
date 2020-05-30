# Optimus

Optimus is a customized cluster scheduler for deep learning training jobs that targets high job performance and resource efficiency in production clusters. It builds resource-performance models for each job on the go, and dynamically schedules resources to jobs based on job progress and the cluster load to maximize training performance and resource efficiency. The implementation uses MXNet as the distributed training framework and schedules jobs based on Kubernetes.

## Setup

Optimus is not bound to any specific version of the software listed below. You can skip this part if your Kubernetes cluster works.

### Software Environment
(1) Ubuntu 14.04.5 Server 64bit LTS;

(2) HDFS 2.8;

(3) Docker 17.06.0-ce;

(4) Kubernetes 1.7;

(5) NVIDIA Driver version >= 375.66;

(6) CUDA version >= 8.0.61;

(7) CuDNN Library version >= 6.0

See [docs](docs/) for installation guide.

### Container Environment
MXNet GPU container (if the server has NVIDIA GPUs): see [images](images/). You may also build a CPU container.

## Usage
The PS load balance algorithm and code are in [mxnet](mxnet/) and it works on MXNet 1.0. The scheduling code is in [scheduler](scheduler/). 
Before running [experimentor.py](scheduler/experimentor.py), make sure hyper-parameters in [params.py](scheduler/params.py) are correct.

Please use the [images](images/) for running, or you can build your own by copying the [scripts](https://github.com/pengyanghua/optimus/tree/master/images/gpu/scripts) into your own CPU or GPU image. These scripts are for parsing training logs and collecting training speed, loss, accuracy etc. 

All training examples (e.g., image classification) in the paper are from the open source community. Most are from MXNet official [examples](https://github.com/apache/incubator-mxnet/tree/master/example) and you can find how to run these examples (e.g., preparing the training data and starting training) there. The machine translation example is from [sockeye](https://github.com/awslabs/sockeye).

This is a prototype, so  and it may take some time to . Before running the code, please read the scheduler code first to understand how Optimus interacts with k8s. That may save you a lot of time if encounting any bugs.

## More
Read the <a href="https://dl.acm.org/citation.cfm?id=3190517"> Optimus paper </a>  and <a href="https://blog.acolyer.org/2018/06/12/optimus-an-efficient-dynamic-resource-scheduler-for-deep-learning-clusters/">the morning report </a> for details.

Contact yhpeng@cs.hku.hk if you have any questions.
