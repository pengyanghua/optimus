import sys
import Queue
import time
import threading
import params
import math
import numpy as np
from scipy.optimize import curve_fit
import random


class Estimator(object):
    def __init__(self, name, logger):
        self.name = name
        self.logger = logger
        self.exit_event = threading.Event()
        self.existing_jobs = []

    '''
    exit flag
    '''
    def set_exit_flag(self, exit_flag):
        if exit_flag:
            self.exit_event.set()

    '''
    test training speed to get points for curve fitting
    '''
    def _test_placement(self, jobs):
        cur_node_index = 0
        # job_id:placement_list
        placements = dict()

        node_used_cpu_list = [0 for i in range(len(params.NODE_LIST))]
        node_used_mem_list = [0 for i in range(len(params.NODE_LIST))]
        node_used_gpu_list = [0 for i in range(len(params.NODE_LIST))]
        node_used_bw_list = [0 for i in range(len(params.NODE_LIST))]

        for job in jobs:
            placements[job.id] = []
            for i in range(job.num_ps): # place each bundle
                # random placement
                cpu_req = job.worker_cpu + job.ps_cpu
                mem_req = job.worker_mem + job.ps_mem
                bw_req = job.worker_bw + job.ps_bw
                gpu_req = job.worker_gpu

                # check whether resources are sufficient
                for i in range(len(params.NODE_LIST)):
                    node_index = (cur_node_index + i) % len(params.NODE_LIST)
                    suff_resr = True
                    if node_used_cpu_list[node_index] + cpu_req > params.CPU_PER_NODE or \
                                            node_used_mem_list[node_index] + mem_req > params.MEM_PER_NODE or \
                                            node_used_bw_list[node_index] + bw_req > params.BW_PER_NODE or \
                                            node_used_gpu_list[node_index] + gpu_req > params.GPU_PER_NODE:
                        suff_resr = False
                        continue

                    if suff_resr == True:
                        # node node_index has enough resources
                        break

                if suff_resr:
                    node_used_cpu_list[node_index] += cpu_req
                    node_used_mem_list[node_index] += mem_req
                    node_used_bw_list[node_index] += bw_req
                    node_used_gpu_list[node_index] += gpu_req

                    # placement
                    if job.id in placements:
                        placements[job.id].append(params.NODE_LIST[node_index])
                    else:
                        placements[job.id] = [params.NODE_LIST[node_index]]
                else:
                    break

                # cur_node_index = (node_index + 1) % len(params.NODE_LIST)  # update index for next job

        return placements

    def _run(self, job, placement):
        # set placement
        self.logger.debug(self.name + ":: " + job.name + ", num_ps: " + str(job.num_ps) + ", num_worker: " + str(
            job.num_worker) + ", placement: " + str(placement))
        job.num_ps = len(placement)
        job.worker = job.num_ps
        job.set_ps_placement(placement)
        job.set_worker_placement(placement)
        # start job
        job.start()

    def test_speed(self, new_jobs):
        self.logger.debug(self.name + ":: " + "start testing training speed for " + str(len(new_jobs)) + " jobs...")

        tic = time.time()

        # little improvement: if two jobs are of the same type, they can reuse the training speed points
        for existing_job in self.existing_jobs:
            for new_job in new_jobs:
                if existing_job.workload_id == new_job.workload_id: # same type of job
                    for key, value in existing_job.training_speeds.items():
                        if key in new_job.training_speeds: # simply average
                            new_job.training_speeds[key] = (new_job.training_speeds[key] + value) / 2
                        else:
                            new_job.training_speeds[key] = value
        self.existing_jobs += new_jobs
        counter = 0
        while True:
            q = Queue.PriorityQueue()
            # less test speed, higher priority
            collected = True
            for job in new_jobs:
                if len(job.training_speeds) < 8: # collect 8 points
                    collected = False
                    break

            if collected:
                # all job has collected 5 points
                self.logger.info("No need to test speed, all jobs are known workload.")
                break
            else:
                # at least one job does not have 5 speed points
                for job in new_jobs:
                    q.put((len(job.training_speeds), job))

            sorted_jobs = []
            while not q.empty():
                num_speed_points, job = q.get()
                if num_speed_points >= 10:
                    continue
                # determine number of ps and number of worker
                while True:
                    job.num_worker = random.randint(1,10)
                    job.num_ps = job.num_worker
                    if (job.num_ps, job.num_worker) in job.training_speeds: # avoid repetition
                        # will cause infinite loop if job already has 10 points
                        continue
                    else:
                        sorted_jobs.append(job)
                        break

            counter += 1
            self.logger.debug(self.name + ":: " + "No." + str(counter) + " time, " + "collecting speed points...")

            placements = self._test_placement(sorted_jobs)
            running_jobs = []
            threads = []
            for job in sorted_jobs:
                placement = placements[job.id]
                if len(placement) == job.num_ps:
                    running_jobs.append(job)
                    thread = threading.Thread(target=self._run, args=(job, placement,))
                    thread.start()
                    threads.append(thread)
                    # multiple concurrent job startings may cause congestion
                    time.sleep(3)
                else:
                    self.logger.debug(self.name + ":: " + job.name + " does not get resources to test speed")

            for thread in threads:
                thread.join()

            # sleep one minute to get training speed (better 5 mins, but may cost much time)
            if len(running_jobs) > 0:
                if self.exit_event.wait(60*3):
                    sys.exit()

            # read training speed, if no, sleep more
            for job in running_jobs:
                flag = True
                while flag:
                    speed_list = job.get_training_speed()
                    if min(speed_list) > 0:
                        job.training_speeds[(job.num_ps, job.num_worker)] = sum(speed_list) / int(job.tot_batch_size)  # batches/second
                        job.delete(True)
                        flag = False
                    else:
                        self.logger.debug(self.name + ":: " + "did not get speed from job " + job.name + " " +  str(speed_list) + ", sleep and try again later.")
                        if self.exit_event.wait(10):
                            sys.exit()

            for job in new_jobs:
                self.logger.debug(self.name + ":: " + job.name + ": " + str(job.training_speeds))

        toc = time.time()
        self.logger.info(self.name + ":: time cost of collecting speed points: " + '%.3f'%(toc - tic) + " seconds")
        # clear modifications
        for job in new_jobs:
            job.num_ps = 0
            job.set_ps_placement([])
            job.num_worker = 0
            job.set_worker_placement([])
        return


    '''
    --------------Below is completion epoch estimation----------------
    '''

    def __loss_fit_func(x, a, b, c):
        return 1/(a*(x)+b) + c

    def _loss_curve_fitting(self, epochs_arr, losses_arr):
        param_bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

        # assign different weights to points, default sigma is ones
        sigma = np.ones(len(epochs_arr))
        NUM_SEGMENTS = 3
        for i in range(len(epochs_arr)):
            exp = int(math.floor(i/(math.ceil(1.0*len(epochs_arr)/NUM_SEGMENTS))))
            sigma[i] /= 4 ** exp

        params = curve_fit(self.__loss_fit_func, epochs_arr, losses_arr, sigma=np.array(sigma), absolute_sigma=False,
                           bounds=param_bounds)
        return params[0]

    def est_epoch(self, job):
        if job.num_epochs < sys.maxint:
            return job.num_epochs

        existing_epochs = []
        for existing_job in self.existing_jobs:
            if existing_job.workload_id == job.workload_id: # same type of job
                if existing_job.num_epochs < sys.maxint:
                    existing_epochs.append(existing_job.num_epochs)

        if len(existing_epochs) > 0:
            # training epoch is already specified
            return int(sum(existing_epochs) / len(existing_epochs))
        else:
            # we need to estimate the number of required epochs
            if len(job.val_losses) >= 3:
                epoch_list = []
                loss_list = []
                for epoch, loss in job.val_losses.items():
                    epoch_list.append(epoch)
                    loss_list.append(loss)

                # we do not need curve fitting each time, can be further optimized in future
                # also, we can get loss data from previous jobs, optimized in future
                try:
                    [a, b, c] = self._loss_curve_fitting(np.array(epoch_list), np.array(loss_list))  # could throw exception since the loss may not descend at the beginning
                except Exception as e:
                    print "loss curve fitting error: ", e
                    return -1
                # if loss does not change a lot for a certain period, converge.
                epoch = max(0, int(job.progress) - params.LOSS_LITTLE_CHANGE_EPOCH_NUM)
                fitted_losses = []
                while True:
                    fitted_losses.append(self.__loss_fit_func(epoch, a, b, c))
                    flag = True
                    if len(fitted_losses) >= params.LOSS_LITTLE_CHANGE_EPOCH_NUM:
                        for i in reversed(range(params.LOSS_LITTLE_CHANGE_EPOCH_NUM)):
                            if fitted_losses[epoch - i] - fitted_losses[epoch] > params.LOSS_CONVERGENCE:
                                flag = False
                                break
                    if not flag:
                        epoch += 1
                        if epoch > 100:  # each job must have at most 100 epochs
                            return -1
                    else:
                        return epoch
            else:
                return -1


    '''
    --------------Below is speed estimation------------
    '''

    def __async_speed_fit_func(X, a, b, c, d):
        p, w = X
        return w / (a + b * w / p + c * w + d * p)

    # async curve fitting to get a,b,c
    def _async_speed_curve_fitting(self, ps_arr, worker_arr, speed_arr):
        param_bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
        sigma = np.ones(len(ps_arr))
        try:
            params = curve_fit(self.__async_speed_fit_func, (ps_arr, worker_arr), speed_arr, sigma=np.array(sigma), absolute_sigma=False, bounds=param_bounds)
            return params[0]
        except Exception as e:
            self.logger.error(str(e))

    def __sync_speed_fit_func(self, X, a, b, c, d, e):
        p, w, batch_size = X
        return 1 / (a * batch_size / w + b + c * w / p + d * w + e * p)

    # curve fitting to get a,b,c
    def _sync_speed_curve_fitting(self, ps_arr, worker_arr, batch_arr, speed_arr):
        param_bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])
        sigma = np.ones(len(ps_arr))
        try:
            params = curve_fit(self.__sync_speed_fit_func, (ps_arr, worker_arr, batch_arr), speed_arr, sigma=np.array(sigma), absolute_sigma=False, bounds=param_bounds)
            return params[0]
        except Exception as e:
            self.logger.error(self.name + ":: " + "curve fitting error, " + str(e))

    def est_speed(self, job, num_ps, num_worker):
        """Give the number of ps and the number of worker, predict the training speed.
        Use the real one if already exists in the dict
        """
        if (num_ps, num_worker) in job.training_speeds:
            return job.training_speeds[(num_ps, num_worker)]
        else:
            # do training speed curve fitting here
            if 'async' in job.kv_store:
                if len(job.training_speeds) >= 4:
                    # do not need curve fitting each time, can be further optimized. future work
                    ps_list = []
                    worker_list = []
                    speed_list = []
                    for key, value in job.training_speeds.items():
                        (ps, worker) = key
                        ps_list.append(float(ps))
                        worker_list.append(float(worker))
                        speed_list.append(value)
                    params = self._async_speed_curve_fitting(np.array(ps_list), np.array(worker_list), np.array(speed_list))
                    if params is None:
                        self.logger.error(self.name+":: " + job.name + " " + str((num_ps, num_worker)) + " speed estimation error")
                        return -1
                    else:
                        [a, b, c, d] = params
                        est_speed = self.__async_speed_fit_func((num_ps, num_worker), a, b, c, d)
                        return est_speed
                else:
                    return -1
            elif 'sync' in job.kv_store:
                if len(job.training_speeds) >= 5:
                    ps_list = []
                    worker_list = []
                    speed_list = []
                    for key, value in job.training_speeds.items():
                        (ps, worker) = key
                        ps_list.append(float(ps))
                        worker_list.append(float(worker))
                        speed_list.append(value)
                    batch_size_list = [float(job.tot_batch_size) for i in range(len(ps_list))]
                    params = self._sync_speed_curve_fitting(np.array(ps_list), np.array(worker_list),
                                                             np.array(batch_size_list), np.array(speed_list))
                    if params is None:
                        self.logger.error(self.name + ":: " + job.name + " " + str((num_ps, num_worker)) + " speed estimation error")
                        return -1
                    else:
                        [a, b, c, d, e] = params
                        est_speed = self.__sync_speed_fit_func((num_ps, num_worker, float(job.tot_batch_size)), a, b, c, d, e)
                        return est_speed
                else:
                    return -1
