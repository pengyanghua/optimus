import time
import os
import threading
import random
import yaml
import numpy

import params
import jobrepo
from job import Job


class Generator(object):
    def __init__(self, name, logger, hub_queue, timer):
        self.name = name
        self.logger = logger
        self.hub_queue = hub_queue
        self.timer = timer

        self.exit_flag = False
        self.job_dict = dict()

        self._generate()  # generate jobs
        self.submitter = threading.Thread(target=self._submit, args=())
        self.submitter.start()

    def set_exit_flag(self, exit_flag):
        self.exit_flag = exit_flag

    def _generate(self):
        tic = time.time()

        job_id = 1
        random.seed(params.RANDOM_SEED)	# make each run repeatable
        numpy.random.seed(params.RANDOM_SEED)
        accum_t = 0
        cwd = os.getcwd() + '/'

        for i in xrange(params.TOT_NUM_JOBS):
            if params.JOB_DISTRIBUTION == "uniform":
                # uniform randomly choose one
                index = random.randint(0,len(jobrepo.job_repos)-1)
                (type, model) = jobrepo.job_repos[index]
                job = Job(job_id, type, model, index, cwd, self.logger)
                jobrepo.set_config(job)

            # randomize job arrival time
            if params.JOB_ARRIVAL == "uniform":
                t = random.randint(1, params.T)  # clock start from 1
                job.arrival_slot = t
            if job.arrival_slot in self.job_dict:
                self.job_dict[job.arrival_slot].append(job)
            else:
                self.job_dict[job.arrival_slot] = [job]

            job_id += 1

        toc = time.time()
        self.logger.debug(self.name + ":: " + "has generated " + str(job_id-1) + " jobs")
        self.logger.debug(self.name + ":: " + "time to generate jobs: " + '%.3f' % (toc - tic) + " seconds.")

    def _submit(self):
        # put jobs into queue
        t = self.timer.get_clock()
        counter = 0
        while not self.exit_flag:
            self.logger.info(
                "\n-------*********--------" + "starting timeslot " + str(t) + "..." + "--------*********-------")
            if t in self.job_dict:
                for job in self.job_dict[t]:
                    msg = (t, 'generator', 'scheduler', 'submission', job)
                    self.hub_queue.put(msg)  # enqueue jobs at the beginning of each time slot
                    job.arrival_time = time.time()
                    counter += 1

            # notify the scheduler that all jobs in this timeslot have been submitted
            msg = (t, 'generator', 'scheduler', 'submission', None)
            self.hub_queue.put(msg)
            while t == self.timer.get_clock():
                time.sleep(params.MIN_SLEEP_UNIT)
            t = self.timer.get_clock()
            if t is None:
                break

        self.logger.debug(self.name + ":: " + "has submitted " + str(counter) + " jobs")
        self.logger.debug(self.name + ":: " + self.name + " has exited.")
