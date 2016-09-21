#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import Queue
import random
import time


class Producter(threading.Thread):
    """生产者线程"""
    def __init__(self, t_name, queue):
        self.queue = queue
        threading.Thread.__init__(self, name=t_name)

    def run(self):
        for i in range(10):
            randomnum = random.randint(1, 99)
            self.queue.put(randomnum)
            print 'put num in Queue %s' %  randomnum
            time.sleep(1)

        print 'put queue done'


class ConsumeEven(threading.Thread):
    """奇数消费线程"""
    def __init__(self, t_name, queue):
        self.queue = queue
        threading.Thread.__init__(self, name=t_name)

    def run(self):
        while True:
            try:
                queue_val = self.queue.get(1, 3)
            except Exception, e:
                print e
                break;

            if queue_val % 2 == 0:
                print 'Get Even Num %s ' % queue_val
            else:
                self.queue.put(queue_val)


q = Queue.Queue()
pt = Producter('producter', q)
ce = ConsumeEven('consumeeven', q)
ce.start()
pt.start()
pt.join()
ce.join()
