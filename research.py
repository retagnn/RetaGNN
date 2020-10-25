
from test import DataCollector
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, default=5)
parser.add_argument('--H', type=int, default=3)
parser.add_argument('--topk', type=int, default=20)

config = parser.parse_args()
datacollector = DataCollector(config)
uid2locid_time,locid2detail,_,_,_,_,_,_,_,_,_,_,v2vc = datacollector.main(save1=True,save2=True)



window_size = 10

uid_list = list(set(uid2locid_time.keys()))
vc_list = list(set(v2vc.values()))
vc2num = dict()
for i in range(len(vc_list)):
    vc2num[vc_list[i]] = 0

AMSE_list = list()
vc_flatten_list = list()
for i in range(len(uid_list)):
    vc_flatten_list.append([])
    locid_time_ = uid2locid_time[uid_list[i]]
    for j in range(len(locid_time_)):
        detail_ = locid2detail[locid_time_[j][0]]
        vc_ = detail_[-1]
        vc_flatten_list[i].append(vc_)

for i in range(len(vc_flatten_list)):
    vc_i_list_ = vc_flatten_list[i]
    window_num_ = len(vc_i_list_) - (2*window_size) + 1
    mse_list_ = list()
    for j in range(window_num_):

        new_vc2num_first = dict()
        for i in range(len(vc_list)):
            new_vc2num_first[vc_list[i]] = 0
        new_vc2num_second = dict()
        for i in range(len(vc_list)):
            new_vc2num_second[vc_list[i]] = 0

        first_ = vc_i_list_[j:j+window_size]
        second_ = vc_i_list_[j+window_size:j+(2*window_size)]
        for k in range(len(first_)):
            if first_[k] in new_vc2num_first:
                new_vc2num_first[first_[k]] +=1
            else:
                print('uuu')
        for k in range(len(second_)):
            if second_[k] in new_vc2num_second:
                new_vc2num_second[second_[k]] +=1
            else:
                print('jjjj')
        mse_ = list()
        for k in range(len(vc_list)):
            first_num_ = new_vc2num_first[vc_list[k]]
            second_num_ = new_vc2num_second[vc_list[k]]
            mse_k = abs(first_num_-second_num_)
            mse_.append(mse_k)
        mse_ = sum(mse_)
        mse_list_.append(mse_)
    amse_ = sum(mse_list_)/len(mse_list_)
    AMSE_list.append(amse_)

print(AMSE_list)
plt.hist(AMSE_list)
plt.savefig("AMSE.png")
plt.show()

        









