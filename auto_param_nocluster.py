import os,sys

cluster_num = [5,10,15,20,25]
component_num = [10,20,30]
window_num = [20, 50, 100,200]

train_f='features_norm_modified.txt'
test_f='features_c1_period_modified.txt'
dir_f = 'c1'
i=10
for j in component_num:
    for k in window_num:
        command = 'python VNF_behavior_individual.py ' + str(i) + ' ' + str(j) + ' ' + str(k) + ' ' + train_f + ' ' + test_f + ' ' + dir_f
        os.system(command)

 
