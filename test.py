

'''
ny.checkin
mid,uid,time,lat,lng,locid
1152544297120748018,7359659,2015-12-31 21:06:48-05:00,40.755269694,-73.980860525,39955188

ny.vname
locid,4sqid,locname
14621096,4f1c1b9fe4b0f67a9769ffce,Christian Pentecostal Church

ny.vcat
locid,catid,rating,likes,cicnt,tipcnt,usercnt
14621096,7032,100.0,4,189,1,55


ny.demo
uid,gender,conf_gender,race,conf_race,age,range_age
35113472,2,97.3803,1,67.4674,31,5
'''

import pickle
import numpy as np
import random

class DataCollector:
    def __init__(self,config):
        self.file_path = '/home/hsucheng/DRS/dataset/RS/'
        self.arg = config
        self.L,self.H,self.topk = self.arg.L,self.arg.H,self.arg.topk
        self.N = 20

    def _load_raw_data_(self):
        '''
        raw data: 1.ny.checkin, 2.ny.vcat, 3.ny.vname
        '''
        self.ny_checkin = open(self.file_path+'ny.checkin','r').readlines()[:500000]
        self.ny_vcat = open(self.file_path+'ny.vcat','r').readlines()
        self.ny_vname = open(self.file_path+'ny.vname','r').readlines()
        self.ny_demo = open(self.file_path+'ny.demo','r').readlines()

    def _ny_demo2dict_(self):
        gender_tocken,race_tocken,age_tocken = 2,3,3
        onehot_len = gender_tocken * race_tocken * age_tocken
        tocken2onehot,index = dict(),0
        for i in range(gender_tocken):
            for j in range(race_tocken):
                for k in range(age_tocken):
                    tocken_ = str(i+1)+'-'+str(j+1)+'-'+str(k+1)
                    onehot_ = list()
                    for h in range(onehot_len):
                        if h == index:
                            onehot_.append(1)
                        else:
                            onehot_.append(0)
                    index += 1
                    tocken2onehot[tocken_] = onehot_
        onehot_list = list(tocken2onehot.values())
        #gender_list,race_list,age_list = list(),list(),list()
        uid2attr = dict()
        for ny_demo_i in self.ny_demo[1:]:
            ny_demo_i = ny_demo_i.split(',')
            uid_,gender_,race_,age_ = ny_demo_i[0],int(ny_demo_i[1]),int(ny_demo_i[3]),int(ny_demo_i[5])
            if age_ <= 20:
                age_ = 1
            elif  age_ > 20 and age_ <= 40:
                age_ = 2
            else:
                age_ = 3
            tocken_ = str(gender_) + '-' + str(race_) + '-' + str(age_)
            onehot_ = tocken2onehot[tocken_]
            if uid_ not in uid2attr:
                onehot_ = random.choice(onehot_list)
                uid2attr[uid_] = onehot_

        uid2attr['nothing'] = onehot_
        return uid2attr

    def _ny_checkin2dict_(self):
        '''
        uid2locid_time : 
        key-> uid_
        value-> [(locid_,time_),(locid_,time_),(locid_,time_),....] 
        locid2detail :
        key-> locid_
        value-> [lat_,lng_]
        '''
        uid2locid_time,locid2detail = dict(),dict()
        for ny_checkin_i in self.ny_checkin[1:]:
            ny_checkin_i = ny_checkin_i.split(',')
            uid_,time_,locid_ = ny_checkin_i[1],ny_checkin_i[2],ny_checkin_i[5].strip('\n')
            lat_,lng_ = ny_checkin_i[3],ny_checkin_i[4]
            if uid_ not in uid2locid_time:
                uid2locid_time[uid_] = list()
            uid2locid_time[uid_] = [(locid_,time_)] + uid2locid_time[uid_] 
            if locid_ not in locid2detail:
                locid2detail[locid_] = [lat_,lng_]
        return uid2locid_time,locid2detail

    def _ny_vname2dict_(self):
        '''
        locid2locname :
        key-> locid_
        value-> locname_
        '''
        locid2locname = dict()
        for ny_vname_i in self.ny_vname[1:]:
            ny_vname_i = ny_vname_i.split(',')
            locid_,locname_ = ny_vname_i[0],ny_vname_i[2]
            if locid_ not in locid2locname:
                locid2locname[locid_] = locname_
        return locid2locname

    def _ny_vcat2dict_(self):
        '''
        locid2catid :
        key-> locid_
        value-> catid_
        '''       
        locid2catid = dict()
        for ny_vcat_i in self.ny_vcat[1:]:
            ny_vcat_i = ny_vcat_i.split(',')
            locid_,catid_ = ny_vcat_i[0],str(int(int(ny_vcat_i[1])*0.001))
            if locid_ not in locid2catid:
                locid2catid[locid_] = catid_
        return locid2catid

    def _locid2detail_Merge_(self,locid2detail,locid2locname,locid2catid,uid2locid_time):
        locid_name_ = list(set(locid2detail.keys()))
        for i in range(len(locid_name_)):
            if locid_name_[i] in locid2locname:
                locname_ = locid2locname[locid_name_[i]]
                locid2detail[locid_name_[i]].append(locname_)
            else:
                locid2detail[locid_name_[i]].append('none_name')
            if locid_name_[i] in locid2catid:
                catid_ = locid2catid[locid_name_[i]]
                locid2detail[locid_name_[i]].append(catid_)
            else:
                locid2detail[locid_name_[i]].append('11')
        return locid2detail
    
    def main_data2dict(self,save=True):
        '''
        uid2locid_time
        key: uid
        value: [(locid,time),(locid,time),...]
        locid2detail
        key: locid
        value: [lat_,lng_,locname_,catid_]
        or value: [lat_,lng_,none_name,11]
        '''
        if save:
            self._load_raw_data_()
            uid2locid_time,locid2detail = self._ny_checkin2dict_()
            locid2locname = self._ny_vname2dict_()
            locid2catid = self._ny_vcat2dict_()
            uid2attr = self._ny_demo2dict_()
            locid2detail = self._locid2detail_Merge_(locid2detail,locid2locname,locid2catid,uid2locid_time)
            ig_main1_data = {
                'uid2locid_time':uid2locid_time,
                'locid2detail':locid2detail,
                'uid2attr':uid2attr,
            }
            file = open('ig_main1_data.pickle', 'wb')
            pickle.dump(ig_main1_data, file)
            file.close()   
        else:
            with open('ig_main1_data.pickle', 'rb') as file:
                ig_main1_data = pickle.load(file)
            uid2locid_time = ig_main1_data['uid2locid_time']
            locid2detail = ig_main1_data['locid2detail']
            uid2attr = ig_main1_data['uid2attr']
        return uid2locid_time,locid2detail,uid2attr

    def _limit_in_seqlen_(self,uid2locid_time):
        old_uid_list_ = list(uid2locid_time.keys())
        for i in range(len(old_uid_list_)):
            locid_time_ = uid2locid_time[old_uid_list_[i]]
            if len(locid_time_) < (self.N + self.L + self.topk):
                del uid2locid_time[old_uid_list_[i]]
            else:
                uid2locid_time[old_uid_list_[i]] = uid2locid_time[old_uid_list_[i]][:self.N + self.L + self.topk]
        return uid2locid_time
     
    def _element2ids_(self,uid2locid_time,locid2detail):
        old_uid_list_ = list(uid2locid_time.keys()) 
        locid_list_,vc_list_ = list(),list()

        for i in range(len(old_uid_list_)):
            locid_time_ = uid2locid_time[old_uid_list_[i]]
            for j in range(len(locid_time_)):
                locid_list_.append(locid_time_[j][0])
                detail_ = locid2detail[locid_time_[j][0]]
                vc_list_.append(detail_[-1])

        locid_list_ = list(set(locid_list_)) 
        vc_list_ = list(set(vc_list_)) 

        ids_ = 0
        self.uid2ids,self.locid2ids,self.vc2ids = dict(),dict(),dict() 

        for i in range(len(locid_list_)):
            self.locid2ids[locid_list_[i]] = ids_
            ids_ +=1
        
        for i in range(len(old_uid_list_)):
            self.uid2ids[old_uid_list_[i]] = ids_
            ids_ +=1
        
        for i in range(len(vc_list_)):
            self.vc2ids[vc_list_[i]] = ids_
            ids_ +=1  
        
        node_num = ids_ 
        return old_uid_list_,locid_list_,node_num

    def _replace_element_with_ids_(self,old_uid_list_,locid_list_,uid2locid_time,locid2detail,uid2attr):
        new_uid2locid_time = dict()
        for i in range(len(old_uid_list_)):
            uid_ids_ = self.uid2ids[old_uid_list_[i]]
            locid_time_ = uid2locid_time[old_uid_list_[i]]
            new_uid2locid_time[uid_ids_] = [(self.locid2ids[locid_time_[j][0]],locid_time_[j][1]) for j in range(len(locid_time_))]
        new_locid2detail = dict()
        for i in range(len(locid_list_)):
            detail_ = locid2detail[locid_list_[i]]
            detail_[-1] = self.vc2ids[detail_[-1]]
            new_locid2detail[self.locid2ids[locid_list_[i]]] = detail_
        new_uid2attr = dict()
        for i in range(len(old_uid_list_)):
            uid_ids_ = self.uid2ids[old_uid_list_[i]]
            if old_uid_list_[i] in uid2attr:
                attr_ = uid2attr[old_uid_list_[i]]
            else:
                attr_ = uid2attr['nothing']  
            new_uid2attr[uid_ids_] = attr_
        return new_uid2locid_time,new_locid2detail,new_uid2attr

    def _seq_data_building_(self,old_uid_list_,uid2locid_time):
        new_uid_list_ = [self.uid2ids[old_uid_list_[i]] for i in range(len(old_uid_list_))]
        user_np,seq_train,seq_test,test_set = list(),list(),list(),list()

        for i in range(len(new_uid_list_)):
            locid_time_ = uid2locid_time[new_uid_list_[i]]
            train_part = locid_time_[:self.N]
            testX_part = locid_time_[self.N:self.N+self.L]
            testY_part = locid_time_[self.N+self.L:]

            for j in range(len(train_part)-self.L-self.H+1):
                train_part_j_ = train_part[j:j+self.L+self.H]
                user_np.append(new_uid_list_[i])
                seq_train.append([train_part_j_[k][0] for k in range(len(train_part_j_))])

            seq_test.append([testX_part[j][0] for j in range(len(testX_part))])
            test_set.append([testY_part[j][0] for j in range(len(testY_part))])
        user_np = np.array(user_np)
        seq_train = np.array(seq_train)
        seq_test = np.array(seq_test)
        return new_uid_list_,user_np,seq_train,seq_test,test_set

    def _edge_building_(self,uid_list_,uid2locid_time,locid2detail):
        u2v,u2vc,v2u,v2vc = dict(),dict(),dict(),dict()
        for i in range(len(uid_list_)):
            locid_time_ = uid2locid_time[uid_list_[i]]
            v_list_,u_vc_list_ = list(),list()
            for j in range(len(locid_time_)):
                locid_ = locid_time_[j][0]
                v_list_.append(locid_)

                if locid_ not in v2u:
                    v2u[locid_] = list()
                v2u[locid_].append(uid_list_[i])

                vc_ = locid2detail[locid_][-1]
                u_vc_list_.append(vc_)
                if locid_ not in v2vc:
                    v2vc[locid_] = vc_
            v_list_ = list(set(v_list_))
            u2v[uid_list_[i]] = v_list_
            u_vc_list_ = list(set(u_vc_list_))
            u2vc[uid_list_[i]] = u_vc_list_
        v2u_keys = list(v2u.keys())
        for i in range(len(v2u_keys)):
            v2u[v2u_keys[i]] = list(set(v2u[v2u_keys[i]]))
        return u2v,u2vc,v2u,v2vc

    def main_datadict2traindata(self,uid2locid_time,locid2detail,uid2attr,save=True):
        if save:
            uid2locid_time = self._limit_in_seqlen_(uid2locid_time)
            old_uid_list_,locid_list_,node_num = self._element2ids_(uid2locid_time,locid2detail)
            uid2locid_time,locid2detail,uid2attr = self._replace_element_with_ids_(old_uid_list_,locid_list_,uid2locid_time,locid2detail,uid2attr)
            uid_list_,user_np,seq_train,seq_test,test_set = self._seq_data_building_(old_uid_list_,uid2locid_time)
            u2v,u2vc,v2u,v2vc = self._edge_building_(uid_list_,uid2locid_time,locid2detail)
            relation_num =  6 #u_vc,u,v,v_vc
            ig_main2_data = {
                'uid2locid_time':uid2locid_time,
                'locid2detail':locid2detail,
                'node_num':node_num,
                'uid_list_':uid_list_,
                'user_np':user_np,
                'seq_train':seq_train,
                'seq_test':seq_test,
                'test_set':test_set,
                'u2v':u2v,
                'u2vc':u2vc,
                'v2u':v2u,
                'v2vc':v2vc,
                'relation_num':relation_num,
                'uid2attr':uid2attr,
            }
            file = open('ig_main2_data.pickle', 'wb')
            pickle.dump(ig_main2_data, file)
            file.close()       
        else:
            with open('ig_main2_data.pickle', 'rb') as file:
                ig_main2_data = pickle.load(file)
            uid2locid_time = ig_main2_data['uid2locid_time']
            locid2detail = ig_main2_data['locid2detail']
            node_num = ig_main2_data['node_num']
            uid_list_ = ig_main2_data['uid_list_']
            user_np = ig_main2_data['user_np']
            seq_train = ig_main2_data['seq_train']
            seq_test = ig_main2_data['seq_test']
            test_set = ig_main2_data['test_set']
            u2v = ig_main2_data['u2v']
            u2vc = ig_main2_data['u2vc']
            v2u = ig_main2_data['v2u']
            v2vc = ig_main2_data['v2vc']
        return uid2locid_time,locid2detail,node_num,relation_num,uid_list_,user_np,seq_train,seq_test,test_set,u2v,u2vc,v2u,v2vc  

    def main(self,save1=True,save2=True):
        #if save1:
        uid2locid_time,locid2detail,uid2attr = self.main_data2dict(save1)
        uid2locid_time,locid2detail,node_num,relation_num,uid_list_,user_np,seq_train,seq_test,test_set,u2v,u2vc,v2u,v2vc = self.main_datadict2traindata(uid2locid_time,locid2detail,uid2attr,save2)
        return uid2locid_time,locid2detail,node_num,relation_num,uid_list_,user_np,seq_train,seq_test,test_set,u2v,u2vc,v2u,v2vc





