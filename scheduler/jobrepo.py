import params


job_repos = [('experiment-imagenet', 'resnet-50'), ('experiment-imagenet', 'vgg-16'),
             ('experiment-cifar10', 'resnext-110')]


def set_config(job):
    is_member = False
    for item in job_repos:
        if job.type == item[0] and job.model_name == item[1]:
            is_member = True

    if not is_member:
        raise RuntimeError

    if 'resnet-50' in job.model_name:
        _set_resnet50_job(job)
    elif 'vgg-16' in job.model_name:
        _set_vgg16_job(job)
    elif 'resnext-110' in job.model_name:
        _set_resnext110_job(job)
    else:
        raise RuntimeError


'''
ResNet-50_ImageNet
'''
def _set_resnet50_job(job):
    num_ps = params.DEFAULT_NUM_PS
    num_worker = params.DEFAULT_NUM_WORKER
    ps_cpu = 3
    ps_mem = 9
    ps_bw = 0
    worker_cpu = 2
    worker_mem = 8
    worker_gpu = 1
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'xxx'
    script = '/init.sh'

    # must end with /, save everything including training data, validation data,
    # training log and training model into this dir
    work_dir = '/mxnet/example/image-classification/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)

    prog = 'python train_imagenet.py --network resnet --num-layers 50 --disp-batches 5 --num-epochs 100 --data-train /data/imagenet-train.rec'
    kv_store = 'dist_sync'
    prog += ' --kv-store ' + kv_store
    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])

    job.set_train(prog=prog, batch_size=32, kv_store=kv_store, scale_bs=True)
    hdfs_data = ['/k8s-mxnet/imagenet/imagenet-train.rec']
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/imagenet/'
    job.set_data(hdfs_data=hdfs_data, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=True)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')


'''
VGG-16_ImageNet
'''
def _set_vgg16_job(job):
    num_ps = params.DEFAULT_NUM_PS
    num_worker = params.DEFAULT_NUM_WORKER
    ps_cpu = 4
    ps_mem = 10
    ps_bw = 0
    worker_cpu = 2
    worker_mem = 10
    worker_gpu = 1
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'xxx'
    script = '/init.sh'

    # must end with /, save everything including training data, validation data,
    # training log and training model into this dir
    work_dir = '/mxnet/example/image-classification/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)

    prog = 'python train_imagenet.py --network vgg --num-layers 16 --disp-batches 2 --num-epochs 100 --data-train /data/imagenet-train.rec'
    kv_store = 'dist_sync'
    prog += ' --kv-store ' + kv_store
    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])
    job.set_train(prog=prog, batch_size=32, kv_store=kv_store, scale_bs=True)
    hdfs_data = ['/k8s-mxnet/imagenet/imagenet-train.rec']
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/imagenet/'
    job.set_data(hdfs_data=hdfs_data, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=True)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')


'''
ResNext-110_Cifar10
'''
def _set_resnext110_job(job):
    num_ps = params.DEFAULT_NUM_PS
    num_worker = params.DEFAULT_NUM_WORKER
    ps_cpu = 3
    ps_mem = 10
    ps_bw = 0
    worker_cpu = 2
    worker_mem = 10
    worker_gpu = 1
    worker_bw = 0

    job.set_ps_resources(num_ps, ps_cpu, ps_mem, ps_bw)
    job.set_worker_resources(num_worker, worker_cpu, worker_mem, worker_bw, worker_gpu)

    image = 'xxx'
    script = '/init.sh'

    # must end with /, save everything including training data, validation data,
    # training log and training model into this dir
    work_dir = '/mxnet/example/image-classification/data/'
    host_workdir_prefix = '/data/k8s-workdir/experiment/'
    job.set_container(image, script, work_dir, host_workdir_prefix)

    prog = 'python train_cifar10.py --network resnext --num-layers 110 --disp-batches 25 \
    --num-epochs 100 --data-train /data/cifar10-train.rec --data-val /data/cifar10-val.rec'
    kv_store = 'dist_sync'
    prog += ' --kv-store ' + kv_store
    if worker_gpu > 0:
        prog += " --gpus" + " " + ",".join([str(i) for i in range(int(worker_gpu))])
    job.set_train(prog=prog, batch_size=128, kv_store=kv_store, scale_bs=True)
    hdfs_data = ['/k8s-mxnet/cifar10/cifar10-train.rec']
    data_dir = '/data/'
    host_data_dir = '/data/mxnet-data/cifar10/'
    job.set_data(hdfs_data=hdfs_data, data_dir=data_dir, host_data_dir=host_data_dir, data_mounted=True)
    job.set_mxnet(kv_store_big_array_bound=1000 * 1000, ps_verbose='')


# Add more examples
