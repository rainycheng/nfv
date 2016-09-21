#!/usr/bin/env python
#coding=utf8

import os,sys,time

for i in range(0,7):
   time.sleep(1)
   print i
   f = open('/home/stack/testwrite','a')
   f.write('hello\n')
   f.write('world\n')
   f.close()
