set -x

# neural machine translation
echo "prepare environment for neural machine translation example..."

apt-get install -y python3-dev python3-setuptools

wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

cp -r /mxnet/python /mxnet/python3 && cd /mxnet/python3/ && python3 setup.py build && python3 setup.py install

mkdir -p /mxnet/example/nmt

#pip3 install sockeye
cd /
wget https://raw.githubusercontent.com/awslabs/sockeye/master/requirements.gpu-cu80.txt
pip3 install sockeye --no-deps -r requirements.gpu-cu80.txt
rm requirements.gpu-cu80.txt
