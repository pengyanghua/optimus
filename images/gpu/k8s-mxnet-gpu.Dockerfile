# build based on official MXNet 0.1 GPU image, preparing all necessary init scripts and training examples.

FROM xxx/mxnet-gpu

MAINTAINER xxx

# parameter load balancing, the modificatin is outdated for mxnet 1.0.0
# COPY scripts/model.py /mxnet/python/mxnet/
# COPY scripts/base_module.py /mxnet/python/mxnet/module/
# COPY scripts/kvstore_dist.h /mxnet/src/kvstore
# COPY scripts/kvstore_dist_server.h /mxnet/src/kvstore/
# COPY scripts/kv_app.h /mxnet/ps-lite/include/ps/

# image-classification
COPY scripts/train_mnist.py /mxnet/example/image-classification/
COPY scripts/train_cifar10.py /mxnet/example/image-classification/
COPY scripts/train_imagenet.py /mxnet/example/image-classification/
COPY scripts/fit.py /mxnet/example/image-classification/common/
COPY scripts/data.py /mxnet/example/image-classification/common/
RUN mkdir -p /mxnet/example/image-classification/data

# cnn text classification
COPY scripts/text_cnn.py /mxnet/example/cnn_text_classification/
COPY scripts/data_helpers.py /mxnet/example/cnn_text_classification/
RUN mkdir -p /mxnet/example/cnn_text_classification/data

# word vector DSSM
COPY scripts/wordvec_subwords_dist.py /mxnet/example/nce-loss/
RUN mkdir -p /mxnet/example/nce-loss/data

# speech recognition, need to recompile mxnet
COPY scripts/deepspeech2.sh /
RUN mkdir -p /mxnet/example/speech_recognition/data
RUN chmod +x /deepspeech2.sh && /bin/bash /deepspeech2.sh

# recomplile with INTEL_MKL and CUDNN
ENV BUILD_OPTS "USE_OPENCV=1 USE_BLAS=openblas USE_DIST_KVSTORE=1 USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 USE_MKL2017=1 USE_MKL2017_EXPERIMENTAL=1 MKLML_ROOT=/mxnet/mkl"
ENV LD_LIBRARY_PATH /mxnet/mkl/lib:$LD_LIBRARY_PATH

RUN cd mxnet && make clean && \
    make -j $(nproc) $BUILD_OPTS && \
    cd python && python2 setup.py build && python2 setup.py install

# neural machine translation, need to recompile python wrapper
RUN mkdir -p /mxnet/example/nmt/data && cd /mxnet/example/nmt && git clone https://github.com/awslabs/sockeye.git
COPY scripts/nmt.sh /mxnet/example/nmt/
RUN  chmod +x /usr/bin/nvidia-smi && chmod +x /mxnet/example/nmt/nmt.sh && /bin/bash /mxnet/example/nmt/nmt.sh

# scripts
COPY scripts/* /

CMD sleep 1000000000
