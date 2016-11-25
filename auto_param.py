import os,sys

cluster_num = [5,10,15,20,25]
component_num = [10,20,30]
window_num = [20, 50, 100,200]

train_f='new_norm_small.txt'
test_f='new_reg_c2.txt'
dir_f = 'c2'
for i in cluster_num:
    for j in component_num:
        for k in window_num:
            command = 'python VNF_behavior.py ' + str(i) + ' ' + str(j) + ' ' + str(k) + ' ' + train_f + ' ' + test_f + ' ' + dir_f
            os.system(command)

 
