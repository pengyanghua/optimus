import Queue
import time
import sys
import threading
import params
from estimator import Estimator


class UTIL_Scheduler(object):
    def __init__(self, name, logger, scheduler_queue, hub_queue, timer):
        self.name = name  # e.g., 'UTIL'
        self.logger = logger
        self.scheduler_queue = scheduler_queue
        self.hub_queue = hub_queue
        self.timer = timer

        self.estimator = Estimator("estimator", self.logger)

        self.cluster_num_cpu = None
        self.cluster_num_mem = None
        self.cluster_num_bw = None
        self.cluster_num_gpu = None
        self.cluster_used_cpu = 0
        self.cluster_used_mem = 0
        self.cluster_used_gpu = 0
        self.cluster_used_bw = 0
        self._set_cluster_config()

        self.queueing_jobs = Queue.PriorityQueue()
        self.uncompleted_jobs = []
        self.completed_jobs = []
        self.cur_ts_completed_jobs = []
        self.not_ready_jobs = set()

        self.exit_flag = False

        self.msg_handler = threading.Thread(target=self._msg_handle, args=())
        self.msg_handler.start()
        self.scaling_overhead = 0
        self.testing_overhead = 0

    def set_exit_flag(self, exit_flag):
        self.exit_flag = exit_flag
        self.estimator.set_exit_flag(exit_flag)

    def _set_cluster_config(self):
        cluster_num_nodes = len(params.NODE_LIST)
        cpu_per_node = params.CPU_PER_NODE
        mem_per_node = params.MEM_PER_NODE
        bw_per_node = params.BW_PER_NODE
        gpu_per_node = params.GPU_PER_NODE
        self.cluster_num_cpu = cluster_num_nodes * cpu_per_node
        self.cluster_num_mem = cluster_num_nodes * mem_per_node
        self.cluster_num_bw = cluster_num_nodes * bw_per_node
        self.cluster_num_gpu = cluster_num_nodes * gpu_per_node

    def _msg_handle(self):
        while not self.exit_flag:
            try:
                (t, src, dest, type, job) = self.scheduler_queue.get(False)
            except:
                continue
            self.logger.debug(self.name + ":: " + str((t, src, dest, type, job)))
            assert t == self.timer.get_clock()
            assert dest == "scheduler"

            if type == "submission" and src == "generator":
                if job is None:
                    # generator has finished the timeslot
                    self._schedule()
                else:
                    job.status = 'queueing'
                    # priority queue based on arrival time
                    self.queueing_jobs.put((job.arrival_time, job))
                    if job not in self.uncompleted_jobs:
                        self.uncompleted_jobs.append(job)
                    else:
                        raise RuntimeError
            elif type == "completion" and src == "progressor":
                if job is None:
                    # progressor has finished the timeslot
                    self._delete()
                else:
                    self.cur_ts_completed_jobs.append(job)
            elif type == "completion" and src == "statsor":
                if job is None:
                    # statsor finishes, start next timeslot
                    self._start_next_ts()
                else:
                    raise RuntimeError
        self.logger.debug(self.name + ":: " + self.name + " has exited.")

    def __update_util_queue(self, job, util_queue):
        # compute utility
        # allocate 1 ps or 1 worker each time.
        # sometimes can allocate multiple ps or worker for optimization, to avoid stuck in local optimal.
        end_epoch = self.estimator.est_epoch(job)
        if end_epoch <= 0:
            # error when estimating epoch
            end_epoch = job.progress + 20

        rem_epoch = end_epoch - job.progress  # the rem_epoch is negative if estimated epoch return -1
        est_speed = self.estimator.est_speed(job, job.num_ps, job.num_worker)
        self.logger.debug("estimated speed: " + str(est_speed))
        if est_speed <= 0:
            self.not_ready_jobs.add(job)
            return
        rem_time = rem_epoch / est_speed

        est_speed = self.estimator.est_speed(job, job.num_ps + 1, job.num_worker)
        if est_speed <= 0:
            self.not_ready_jobs.add(job)
            return
        ps_rem_time = rem_epoch / est_speed
        resource_reqs = (job.ps_cpu, job.ps_mem, job.ps_bw)
        shares = (1.0 * job.ps_cpu / self.cluster_num_cpu, 1.0 * job.ps_mem / self.cluster_num_mem,
                  1.0 * job.ps_bw / self.cluster_num_bw)
        dom_res = shares.index(max(shares))
        ps_util = (rem_time - ps_rem_time)/ resource_reqs[dom_res]

        # if add worker 1
        est_speed = self.estimator.est_speed(job, job.num_ps, job.num_worker + 1)
        if est_speed <= 0:
            self.not_ready_jobs.add(job)
            return
        worker_rem_time = rem_epoch / est_speed
        resource_reqs = (job.worker_cpu, job.worker_mem, job.worker_bw, job.worker_gpu)
        shares = (1.0 * job.worker_cpu / self.cluster_num_cpu, 1.0 * job.worker_mem / self.cluster_num_mem,
                  1.0 * job.worker_bw / self.cluster_num_bw, 1.0 * job.worker_gpu / self.cluster_num_gpu)
        dom_res = shares.index(max(shares))

        worker_util = (rem_time - worker_rem_time) / resource_reqs[dom_res]
        if ps_util >= worker_util:
            # negative util since we prioritize max util
            util_queue.put((-ps_util, job.arrival_time, job, "ps"))
        else:
            util_queue.put((-worker_util, job.arrival_time, job, "worker"))

    def __check_cluster_resource_full(self, cpu_req, mem_req, bw_req=0, gpu_req=0):
        # check whether cluster resources are sufficient
        suff_resr = True
        if self.cluster_used_cpu + cpu_req > self.cluster_num_cpu or \
                                self.cluster_used_mem + mem_req > self.cluster_num_mem or \
                                self.cluster_used_bw + bw_req > self.cluster_num_bw or \
                                self.cluster_used_gpu + gpu_req > self.cluster_num_gpu:
            suff_resr = False
        return suff_resr

    def __check_node_resource_full(self, node_index, cpu_req, mem_req, bw_req=0, gpu_req=0):
        # check whether resources on the node is full
        suff_resr = True
        if self.node_used_cpu_list[node_index] + cpu_req > params.CPU_PER_NODE or \
                                self.node_used_mem_list[node_index] + mem_req > params.MEM_PER_NODE or \
                                self.node_used_bw_list[node_index] + bw_req > params.BW_PER_NODE or \
                                self.node_used_gpu_list[node_index] + gpu_req > params.BW_PER_NODE:
            suff_resr = False
        return suff_resr

    def __deduct_resr(self, job, task_type, task_num, node_index):
        # minus resources on the node
        if task_type == "ps":
            self.node_used_cpu_list[node_index] += job.ps_cpu * task_num
            self.node_used_mem_list[node_index] += job.ps_mem * task_num
            self.node_used_bw_list[node_index] += job.ps_bw * task_num
        elif task_type == "worker":
            self.node_used_cpu_list[node_index] += job.worker_cpu * task_num
            self.node_used_mem_list[node_index] += job.worker_mem * task_num
            self.node_used_bw_list[node_index] += job.worker_bw * task_num
            self.node_used_gpu_list[node_index] += job.worker_gpu * task_num

    def __add_back_resr(self, job, task_type, task_num, node_index):
        # add resources on the node
        if task_type == "ps":
            self.node_used_cpu_list[node_index] -= job.ps_cpu * task_num
            self.node_used_mem_list[node_index] -= job.ps_mem * task_num
            self.node_used_bw_list[node_index] -= job.ps_bw * task_num
        elif task_type == "worker":
            self.node_used_cpu_list[node_index] -= job.worker_cpu * task_num
            self.node_used_mem_list[node_index] -= job.worker_mem * task_num
            self.node_used_bw_list[node_index] -= job.worker_bw * task_num
            self.node_used_gpu_list[node_index] -= job.worker_gpu * task_num

    def __place(self, jobs):
        tic = time.time()

        # keep track of available resources on each node.
        self.node_used_cpu_list = [0 for i in range(len(params.NODE_LIST))]
        self.node_used_mem_list = [0 for i in range(len(params.NODE_LIST))]
        self.node_used_bw_list = [0 for i in range(len(params.NODE_LIST))]
        self.node_used_gpu_list = [0 for i in range(len(params.NODE_LIST))]

        # sort jobs based on num_ps and num_worker
        job_sort_queue = Queue.PriorityQueue()
        for job in jobs:
            job_sort_queue.put((job.num_ps + job.num_worker, job))

        cpu_avail_queue = Queue.PriorityQueue()
        # sort nodes based on available cpus, since cpu is usually the bottleneck
        for i in range(len(params.NODE_LIST)):
            cpu_avail_queue.put((self.node_used_cpu_list[i], i))

        ps_placements = dict()
        worker_placements = dict()

        while not job_sort_queue.empty():
            task_num, job = job_sort_queue.get()
            # check if node resource can satisfy the job's resource requirements
            cand_place_nodes = []
            while not cpu_avail_queue.empty():
                avail_cpu, node_index = cpu_avail_queue.get()
                cand_place_nodes.append(node_index)

                # try to place the job on cand_place_nodes
                fit_flag = True  # whether these nodes can hold the job
                ps_nodes = []
                ps_already_deduct = False

                for i in range(job.num_ps):
                    # place ps evenly
                    node = cand_place_nodes[i % len(cand_place_nodes)]
                    # check whether resource is enough to place this ps
                    suff_resr = self.__check_node_resource_full(node, job.ps_cpu, job.ps_mem, job.ps_bw)
                    if suff_resr:
                        ps_nodes.append(node)
                        # minus temporary resources
                        self.__deduct_resr(job, "ps", 1, node)
                    else:
                        # since node is already sorted based on resources,
                        # if a larger node can not place the task, the following one can not too
                        fit_flag = False
                        # add the deducted resource back
                        for node in ps_nodes:
                            self.__add_back_resr(job, "ps", 1, node)
                        ps_already_deduct = True
                        break
                worker_nodes = []
                for i in range(job.num_worker):
                    # also place worker evenly
                    node = cand_place_nodes[i % len(cand_place_nodes)]
                    # check whether resource is enough to place this ps
                    suff_resr = self.__check_node_resource_full(node, job.worker_cpu, job.worker_mem, job.worker_bw,
                                                                job.worker_gpu)
                    if suff_resr:
                        worker_nodes.append(node)
                        self.__deduct_resr(job, "worker", 1, node)
                    else:
                        fit_flag = False

                        # add the deducted resource back
                        for node in worker_nodes:
                            self.__add_back_resr(job, "worker", 1, node)
                        if not ps_already_deduct:
                            for node in ps_nodes:
                                self.__add_back_resr(job, "ps", 1,
                                                     node)
                        break

                if fit_flag:
                    ps_placements[job.id] = [params.NODE_LIST[node] for node in ps_nodes]
                    worker_placements[job.id] = [params.NODE_LIST[node] for node in worker_nodes]
                    for node in cand_place_nodes:  # enqueue them back
                        cpu_avail_queue.put((self.node_used_cpu_list[node], node))
                    break
                else:
                    if not cpu_avail_queue.empty():
                        # add one more node to see if the job can be fitted
                        continue
                    else:
                        # have try all nodes, but still can not place, then check if we can place some tasks
                        # and place ps and worker alternatively
                        self.logger.debug("last placed job: " + job.name)
                        ps_nodes = []
                        worker_nodes = []
                        flag_place_ps = True
                        for i in range(job.num_ps + job.num_worker):
                            flag_no_resource = True
                            if flag_place_ps:
                                # place ps task
                                for node in range(len(params.NODE_LIST)):
                                    suff_resr = self.__check_node_resource_full(node, job.ps_cpu, job.ps_mem, job.ps_bw)
                                    if suff_resr:
                                        ps_nodes.append(node)
                                        self.__deduct_resr(job, "ps", 1, node)
                                        flag_no_resource = False
                                        break
                            else:
                                # place worker task
                                for node in range(len(params.NODE_LIST)):
                                    suff_resr = self.__check_node_resource_full(node, job.worker_cpu, job.worker_mem,
                                                                                job.worker_bw, job.worker_gpu)
                                    if suff_resr:
                                        worker_nodes.append(node)
                                        self.__deduct_resr(job, "worker", 1, node)
                                        flag_no_resource = False
                                        break
                            if flag_no_resource:
                                break
                            flag_place_ps = not flag_place_ps  # change to place the other task
                            if len(ps_nodes) >= job.num_ps:  # all ps tasks have been placed
                                flag_place_ps = False
                            if len(worker_nodes) >= job.num_worker:  # all worker tasks have been placed
                                flag_place_ps = True

                        if len(ps_nodes) > 0 and len(worker_nodes) > 0:
                            ps_placements[job.id] = [params.NODE_LIST[node] for node in ps_nodes]
                            job.num_ps = len(ps_placements[job.id])
                            worker_placements[job.id] = [params.NODE_LIST[node] for node in worker_nodes]
                            job.num_worker = len(worker_placements[job.id])
                        else:
                            for node in ps_nodes:
                                self.__add_back_resr(job, "ps", 1, node)
                            for node in worker_nodes:
                                self.__add_back_resr(job, "worker", 1, node)
                        # break the while loop
                        break
        self.logger.debug("used cpu: " + str(self.node_used_cpu_list))
        toc = time.time()
        self.logger.info(self.name + ":: " + "Finish job placement in " + "%.3f" % (toc - tic) + " seconds.")
        return (ps_placements, worker_placements)

    def _schedule(self):
        # first collect speed data points
        new_jobs = []
        while not self.queueing_jobs.empty():
            (arrival_time, job) = self.queueing_jobs.get()
            new_jobs.append(job)

        test_tic = time.time()
        # first estimate speed
        self.estimator.existing_jobs = self.uncompleted_jobs + self.completed_jobs
        self.logger.debug(self.name + ":: " + "newly arrived jobs: " + str(new_jobs))
        self.estimator.test_speed(new_jobs)
        self.logger.debug("FINISH TESTING SPEED FOR NEW JOBS.")
        test_toc = time.time()
        self.testing_overhead += (test_toc - test_tic)

        # UTIL
        tic = time.time()

        # a queue based on job utility
        util_queue = Queue.PriorityQueue()

        self.cluster_used_cpu = 0
        self.cluster_used_mem = 0
        self.cluster_used_bw = 0
        self.cluster_used_gpu = 0

        # allocate each job a worker and a server to avoid starvation
        for job in self.uncompleted_jobs:
            cpu_req = job.worker_cpu + job.ps_cpu
            mem_req = job.worker_mem + job.ps_mem
            bw_req = job.worker_bw + job.ps_bw
            gpu_req = job.worker_gpu

            suff_resr = self.__check_cluster_resource_full(cpu_req, mem_req, bw_req, gpu_req)
            if suff_resr:
                job.num_worker = 1
                job.num_ps = 1
                self.cluster_used_cpu += cpu_req
                self.cluster_used_mem += mem_req
                self.cluster_used_bw += bw_req
                self.cluster_used_gpu += gpu_req
                # compute initial utility
                self.__update_util_queue(job, util_queue)
            else:
                continue

        # allocate resources based on job utility
        while not util_queue.empty():
            (util, job_arrival, job, task_type) = util_queue.get()

            # increasing resource leads to slower speed
            # also, newly arrived jobs have negative utility, how to handle this
            if util > 0:
                # must be negative
                break
            if task_type == "ps":
                cpu_req = job.ps_cpu
                mem_req = job.ps_mem
                bw_req = job.ps_bw
                gpu_req = 0
            elif task_type == "worker":
                cpu_req = job.worker_cpu
                mem_req = job.worker_mem
                bw_req = job.worker_bw
                gpu_req = job.worker_gpu

            # check whether resources are sufficient
            suff_resr = self.__check_cluster_resource_full(cpu_req, mem_req, bw_req, gpu_req)
            if suff_resr:
                # currently no mechanism to reduce resources
                if task_type == "ps":
                    job.num_ps += 1
                elif task_type == "worker":
                    job.num_worker += 1
                self.cluster_used_cpu += cpu_req
                self.cluster_used_mem += mem_req
                self.cluster_used_bw += bw_req
                self.cluster_used_gpu += gpu_req

                self.__update_util_queue(job, util_queue)
            else:
                # no enough resource
                break
        # how to handle not_ready_jobs
        self.logger.debug(self.name + ":: " + "not ready jobs " + str(self.not_ready_jobs))

        # check the scheduling result
        for job in self.uncompleted_jobs:
            self.logger.debug(
                self.name + ":: scheduling results" + " num_ps: " + str(job.num_ps) + " num_worker: " + str(
                    job.num_worker))

        # how to handle remaining resources? Due to error sometimes allocating resource can still increase speed

        toc = time.time()
        self.logger.debug(self.name + ":: " + "scheduling time: " + "%.3f" % (toc - tic) + " seconds.")

        # placement
        ps_placements, worker_placements = self.__place(self.uncompleted_jobs)

        scaling_tic = time.time()
        self.running_jobs = []
        # send message to progress to update job progress
        thread_list = []
        for job in self.uncompleted_jobs:
            if job.id not in ps_placements:
                continue
            ps_placement = ps_placements[job.id]
            worker_placement = worker_placements[job.id]
            if len(ps_placement) > 0 and len(worker_placement) > 0:
                # this may cause many ssh connections on a server and an error "ssh_exchange_identification: Connection closed by remote host"
                # to avoid this error, run 'echo "MaxStartups 100:10:200" | sudo tee -a /etc/ssh/sshd_config && sudo service ssh restart' on the server
                self.running_jobs.append(job)
                thread = threading.Thread(target=self.__run, args=(job, ps_placement, worker_placement,))
                thread.start()
                thread_list.append(thread)
                job.status = 'running'

                # send message to progressor
                msg = (self.timer.get_clock(), 'scheduler', 'progressor', 'running', job)
                self.hub_queue.put(msg)
            else:
                job.status = 'pending'

                # send message to progressor
                msg = (self.timer.get_clock(), 'scheduler', 'progressor', 'pending', job)
                self.hub_queue.put(msg)

        for thread in thread_list:
            thread.join()
        scaling_toc = time.time()
        self.scaling_overhead += (scaling_toc - scaling_tic)
        self.logger.debug(
            self.name + ":: " + "job starting time: " + "%.3f" % (scaling_toc - scaling_tic) + " seconds.")

        # send message to progressor to signal scheduling completion
        msg = (self.timer.get_clock(), 'scheduler', 'progressor', 'done', None)
        self.hub_queue.put(msg)

    def __run(self, job, ps_placement, worker_placement):
        self.logger.debug(self.name + ":: " + job.name + ", num_ps: " + str(job.num_ps) + ", num_worker: " + str(
            job.num_worker) + ", ps placement: " + str(ps_placement) + ", worker placement: " + str(worker_placement))

        # set placement and start job
        # sys.exit()
        job.set_ps_placement(ps_placement)
        job.set_worker_placement(worker_placement)
        job.start()

    def _delete(self):
        for job in self.cur_ts_completed_jobs:
            self.uncompleted_jobs.remove(job)
            self.running_jobs.remove(job)
            self.completed_jobs.append(job)

        self.cur_ts_completed_jobs = []

        delete_tic = time.time()

        # clear existing jobs for next time slot
        for job in self.running_jobs:
            job.delete(True)
        delete_toc = time.time()
        self.scaling_overhead += (delete_toc - delete_tic)
        self.logger.debug(self.name + ":: " + "job shutdown time: " + "%.3f" % (delete_toc - delete_tic) + " seconds.")

        # send message to statsor to get statistics of this timeslot
        msg = (self.timer.get_clock(), 'scheduler', 'statsor', 'control', None)
        self.hub_queue.put(msg)

    def _start_next_ts(self):
        # send message to timer to signal starting next timeslot
        msg = (self.timer.get_clock(), 'scheduler', 'timer', 'control', None)
        self.hub_queue.put(msg)
