# Optimus

Optimus is a customized cluster scheduler for deep learning training jobs that targets high job performance and resource efficiency in production clusters. It builds resource-performance models for each job on the go, and dynamically schedules resources to jobs based on job progress and the cluster load to maximize training performance and resource efficiency. The implementation uses MXNet as the distributed training framework and schedules jobs based on Kubernetes.

## Setup

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
MXNet GPU container (if the server has NVIDIA GPUs): see [images](images/)

## Usage
The PS load balance algorithm and code are in [mxnet](mxnet/). The scheduling code is in [scheduler](scheduler/). 
Before running [experimentor.py](scheduler/experimentor.py), make sure hyper-parameters in [params.py](scheduler/params.py) are correct.

## More
Read the <a href="https://dl.acm.org/citation.cfm?id=3190517"> Optimus paper </a>  and <a href="https://blog.acolyer.org/2018/06/12/optimus-an-efficient-dynamic-resource-scheduler-for-deep-learning-clusters/">the morning report </a> for details.
