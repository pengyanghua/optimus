import time
import threading
import Queue
import params


class Timer(object):
    def __init__(self, name, logger, hub_queue, timer_queue):
        self.name = name
        self.logger = logger
        self.timer_queue = timer_queue

        self.clock = 1
        self.exit_flag = False

        self.msg_handler = threading.Thread(target=self._msg_handle, args=())
        self.msg_handler.start()

    def _msg_handle(self):
        while not self.exit_flag:
            try:
                (t, src, dest, type, job) = self.timer_queue.get(False)
            except:
                continue
            self.logger.debug(self.name + ":: " + str((t, src, dest, type, job)))
            assert t == self.clock
            assert dest == "timer"

            if src == "scheduler" and type == "control":
                # scheduler have finished its slot
                self.clock += 1
                self.logger.debug(self.name + ":: " + "add clock 1, clock: " + str(self.clock))

        self.logger.debug(self.name + ":: " + self.name + " has exited.")

    def set_exit_flag(self, exit_flag):
        self.exit_flag = exit_flag

    def get_clock(self):
        if not self.exit_flag:
            return self.clock
