import time
import threading
import Queue
import params


class Statsor(object):
    def __init__(self, name, logger, statsor_queue, hub_queue, timer, scheduler, progressor):
        self.name = name
        self.logger = logger
        self.statsor_queue = statsor_queue
        self.hub_queue = hub_queue
        self.timer = timer
        self.scheduler = scheduler
        self.progressor = progressor

        self.tic = time.time()
        self.end = None

        self.exit_flag = False

        self.msg_handler = threading.Thread(target=self._msg_handle, args=())
        self.msg_handler.start()

        self.stats_txt = "exp-stats.txt"
        f = open(self.stats_txt, 'w')
        f.close()

    def _msg_handle(self):
        while not self.exit_flag:
            try:
                (t, src, dest, type, job) = self.statsor_queue.get(False)
            except:
                continue
            self.logger.debug(self.name + ":: " + str((t, src, dest, type, job)))
            assert t == self.timer.get_clock()
            assert dest == "statsor"

            if type == "control" and src == "scheduler":
                # signal that the scheduler has finished its timeslot and we can start getting statistics
                self._stats(t)
            else:
                raise RuntimeError

        self.logger.debug(self.name + ":: " + self.name + " has exited.")

    def _stats(self, t):
        self.logger.info(self.name + ":: " + "time slot: " + str(t) + "")
        num_submit_jobs = len(self.scheduler.uncompleted_jobs) + len(self.scheduler.completed_jobs)
        num_completed_jobs = len(self.scheduler.completed_jobs)
        num_uncompleted_jobs = len(self.scheduler.uncompleted_jobs)
        self.logger.info(self.name + ":: " + "submitted jobs: " + str(num_submit_jobs) + ", " +
                         "completed jobs: " + str(num_completed_jobs) + ", " +
                         "uncompleted_jobs: " + str(num_uncompleted_jobs))

        cluster_cpu_util = float('%.3f' % (1.0 * self.scheduler.cluster_used_cpu / self.scheduler.cluster_num_cpu))
        cluster_mem_util = float('%.3f' % (1.0 * self.scheduler.cluster_used_mem / self.scheduler.cluster_num_mem))
        cluster_bw_util = float('%.3f' % (1.0 * self.scheduler.cluster_used_bw / self.scheduler.cluster_num_bw))
        cluster_gpu_util = float('%.3f' % (1.0 * self.scheduler.cluster_used_gpu / self.scheduler.cluster_num_gpu))

        self.logger.info(self.name + ":: " + "CPU utilization: " + '%.3f' % (100.0 * cluster_cpu_util) + "%, " +
                         "MEM utilization: " + '%.3f' % (100.0 * cluster_mem_util) + "%, " +
                         "BW utilization: " + '%.3f' % (100.0 * cluster_bw_util) + "%, " +
                         "GPU utilization: " + '%.3f' % (100.0 * cluster_gpu_util) + "%, "
                         )

        # get total number of running tasks
        tot_num_running_tasks = self.progressor.num_running_tasks

        completion_time_list = []
        completion_slot_list = []
        for job in self.scheduler.completed_jobs:
            completion_time_list.append(job.end_time - job.arrival_time)
            completion_slot_list.append(job.end_slot - job.arrival_slot + 1)
        try:
            avg_completion_time = 1.0 * sum(completion_time_list) / len(completion_time_list)
            avg_completion_slot = sum(completion_slot_list) / len(completion_slot_list)
        except:
            self.logger.debug(self.name + ":: " + "No jobs are finished!!!")
        else:
            self.logger.debug(
                self.name + ":: " + "average completion time (including speed measurement): " + '%.3f' % avg_completion_time + " seconds" + \
                ", average completion slots: " + str(avg_completion_slot))

        stats_dict = dict()
        stats_dict["JOB_SCHEDULER"] = params.JOB_SCHEDULER
        stats_dict["JOB_ARRIVAL"] = params.JOB_ARRIVAL
        stats_dict["JOB_DISTRIBUTION"] = params.JOB_DISTRIBUTION
        stats_dict["timeslot"] = t
        stats_dict["num_submit_jobs"] = num_submit_jobs
        stats_dict["num_completed_jobs"] = num_completed_jobs
        stats_dict["num_uncompleted_jobs"] = num_uncompleted_jobs
        stats_dict["cluster_cpu_util"] = cluster_cpu_util
        stats_dict["cluster_mem_util"] = cluster_mem_util
        stats_dict["cluster_bw_util"] = cluster_bw_util
        stats_dict["cluster_gpu_util"] = cluster_gpu_util
        stats_dict["tot_num_running_tasks"] = tot_num_running_tasks
        if self.scheduler.name == "TETRIS_Scheduler" or self.scheduler.name == "UTIL_Scheduler":
            stats_dict["scaling_overhead"] = self.scheduler.scaling_overhead
        if self.scheduler.name == "TETRIS_Scheduler" or self.scheduler.name == "UTIL_Scheduler":
            stats_dict["testing_overhead"] = self.scheduler.testing_overhead
        if len(completion_time_list) > 0:
            stats_dict["avg_completion_time"] = float('%.3f' % (avg_completion_time))
        else:
            stats_dict["avg_completion_time"] = -1
        try:
            ps_cpu_usage = self.progressor.ps_cpu_occupations
            worker_cpu_usage = self.progressor.worker_cpu_occupations
            stats_dict["ps_cpu_usage"] = ps_cpu_usage
            stats_dict["worker_cpu_usage"] = worker_cpu_usage
        except Exception as e:
            self.logger.debug(self.name + ":: " + str(e))

        toc = time.time()
        runtime = toc - self.tic
        stats_dict["runtime"] = float('%.3f' % (runtime))
        if len(self.scheduler.completed_jobs) == params.TOT_NUM_JOBS:
            self.logger.info(self.name + ":: " + "All jobs are completed!")
            if self.end is None:
                self.end = runtime
            stats_dict["makespan"] = float('%.3f' % (self.end))
        else:
            stats_dict["makespan"] = -1

        with open(self.stats_txt, 'a') as f:
            f.write(str(stats_dict) + "\n")

        msg = (t, 'statsor', 'scheduler', 'completion', None)
        self.hub_queue.put(msg)

    def set_exit_flag(self, exit_flag):
        self.exit_flag = exit_flag
