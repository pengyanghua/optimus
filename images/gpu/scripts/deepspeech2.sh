echo "preparing environment for speech recognition DeepSpeech2..."

sudo pip install tensorboard
sudo pip install soundfile

# warpctc
cd /
git clone https://github.com/baidu-research/warp-ctc && cd warp-ctc && mkdir build && cd build && cmake .. && make && sudo make install

echo "WARPCTC_PATH = /warp-ctc" >> /mxnet/make/config.mk
echo "MXNET_PLUGINS += plugin/warpctc/warpctc.mk" >> /mxnet/make/config.mk

# rebuild mxnet
