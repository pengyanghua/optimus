import time
import threading
import params


class Hub(object):
    def __init__(self, name, logger, hub_queue, timer_queue, scheduler_queue, progressor_queue, statsor_queue):
        self.name = name
        self.logger = logger
        self.hub_queue = hub_queue
        self.timer_queue = timer_queue
        self.scheduler_queue = scheduler_queue
        self.progressor_queue = progressor_queue
        self.statsor_queue = statsor_queue

        self.exit_flag = False

        self.msg_handler = threading.Thread(target=self._msg_handle, args=())
        self.msg_handler.start()

    def _msg_handle(self):
        while not self.exit_flag:
            try:
                msg = self.hub_queue.get(False)  # blocking
            except:
                continue
            (t, src, dest, type, job) = msg
            if dest == "scheduler":
                self.scheduler_queue.put(msg)
            elif dest == "progressor":
                self.progressor_queue.put(msg)
            elif dest == "timer":
                self.timer_queue.put(msg)
            elif dest == "statsor":
                self.statsor_queue.put(msg)
            else:
                raise RuntimeError

        self.logger.debug(self.name + ":: " + self.name + " has exited.")

    def set_exit_flag(self, exit_flag):
        self.exit_flag = exit_flag
