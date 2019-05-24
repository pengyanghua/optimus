import threading
import Queue
import time
import sys
import os

import logger
from optimus_scheduler import UTIL_Scheduler
from generator import Generator
from hub import Hub
from progressor import Progressor
from timer import Timer
from statsor import Statsor
import params


def clear():
    os.system('kubectl delete jobs --all')


# an event-driven experimentor
def main():
    clear()

    tic = time.time()

    my_logger = logger.getLogger('job_scheduling_experimentor', params.LOG_LEVEL)
    my_logger.debug("experimentor" + ":: " + "scheduler: " + params.JOB_SCHEDULER)
    hub_queue = Queue.Queue()  # message hub
    timer_queue = Queue.Queue()
    scheduler_queue = Queue.Queue()
    progressor_queue = Queue.Queue()
    statsor_queue = Queue.Queue()

    # start each module in separate thread
    timer = Timer("timer", my_logger, hub_queue, timer_queue)
    hub = Hub("hub", my_logger, hub_queue, timer_queue, scheduler_queue, progressor_queue, statsor_queue)
    generator = Generator("generator", my_logger, hub_queue, timer)
    if params.JOB_SCHEDULER == "UTIL":
        scheduler = UTIL_Scheduler("UTIL_Scheduler", my_logger, scheduler_queue, hub_queue, timer)
    else:
        raise Exception
    progressor = Progressor("progressor", my_logger, progressor_queue, hub_queue, timer)
    statsor = Statsor("statsor", my_logger, statsor_queue, hub_queue, timer, scheduler, progressor)

    # here determine whether all jobs have been finished
    my_logger.info("experimentor:: sleeping...")
    try:
        while len(scheduler.completed_jobs) < params.TOT_NUM_JOBS:
            time.sleep(params.MIN_SLEEP_UNIT)
    except:
        my_logger.error("experimentor:: detect CTRL+C, exit. ")

    exit_flag = True

    timer.set_exit_flag(exit_flag)
    generator.set_exit_flag(exit_flag)
    scheduler.set_exit_flag(exit_flag)
    progressor.set_exit_flag(exit_flag)

    time.sleep(params.MIN_SLEEP_UNIT*100)  # wait for statsor
    statsor.set_exit_flag(exit_flag)
    hub.set_exit_flag(exit_flag)

    time.sleep(params.MIN_SLEEP_UNIT * 3000)  # wait other thread exit

    my_logger.debug("experimentor:: delete unfinished jobs...")
    thread_list = []
    for job in scheduler.uncompleted_jobs:
        # job.delete(True)
        del_thread = threading.Thread(target=(lambda job=job: job.delete(True)), args=())
        del_thread.start()
        thread_list.append(del_thread)
    for del_thread in thread_list:
        del_thread.join()

    toc = time.time()
    my_logger.info("total experiment time: " + "%.3f"%(toc-tic) + " seconds")

    '''
    if platform.system() == "Linux":
        os.system("sudo pkill -9 python")
    elif platform.system() == "Windows":
        os.system("taskkill /im python.exe /F")
    '''


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print "Description: job scheduling experimentor"
        print "Usage: python experimentor.py"
        sys.exit(1)
    main()
