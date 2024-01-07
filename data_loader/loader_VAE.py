import os
import numpy as np
import pandas as pd
from scipy import sparse
import bottleneck as bn

class DataLoaderVAE(object):
    def __init__(self, args):
        data_path = os.path.join(args.data_dir, args.data_name)
        train_data = pd.read_csv(os.path.join(data_path, 'train.txt'), header=0, sep = '|', names = ['uid', 'sid'])
        test_data = pd.read_csv(os.path.join(data_path, 'test.txt'), header=0, sep = '|', names = ['uid', 'sid'])
        print(test_data)
        train_data['sid'] = train_data['sid'].copy().apply(lambda x : [int(elem) for elem in x.split(' ')])
        test_data['sid'] = test_data['sid'].copy().apply(lambda x : [int(elem) for elem in x.split(' ')])
        
        
        self.train_data = train_data.explode('sid')
        self.test_data = test_data.explode('sid')
        
        self.item_to_their_task_id = pd.read_csv(os.path.join(data_path, 'item_to_task_id.csv'))
        
        
       
        n_users = int(self.train_data['uid'].max())
        n_items = int(self.train_data['sid'].max())
        
        self.n_users = max(n_users, int(self.test_data['uid'].max())) + 1
        self.n_items = max(n_items, int(self.test_data['sid'].max())) + 1
        
        self.task_ids = [int(elem) for elem in args.task_ids.split(',')]
        self.max_tasks_ids = max(self.task_ids)
        self.real_train_data = {}
        self.real_test_data = {}
        if args.vae_using_ufo_space:
            self.get_full_data()
        else:
            self.get_full_data_basic()
        self.items_belong_to_past_tasks = {}
        for index in range(len(self.task_ids)):
            if index > 0:
                items_of_recent_task = self.item_to_their_task_id[self.item_to_their_task_id['task_id'] == self.task_ids[index]]
                items_of_past_task = self.item_to_their_task_id[self.item_to_their_task_id['task_id'] == self.task_ids[index - 1]]
                users_having_interaction_with_past_domain = self.train_data['uid'][self.train_data['sid'].isin(items_of_past_task['new_id'].unique())].unique()
                self.users_interactions_at_past_domain = self.train_data[self.train_data['uid'].isin(users_having_interaction_with_past_domain)]
                self.items_belong_to_past_tasks[index] = items_of_recent_task['new_id'][(items_of_recent_task['new_id'].isin(items_of_past_task['new_id'].unique())) | (items_of_recent_task['new_id'].isin(self.users_interactions_at_past_domain['sid'].unique()))].values
            
    def load_tr_te_data(self, train_data_for_train, train_data, train_data_for_test, test_data):
        def get_new_id(data):
            raw_uid_to_new_uid = {value: index for index, value in enumerate(data['uid'].unique())}
            return raw_uid_to_new_uid
        raw_uid_to_new_uid_train_data = get_new_id(train_data)
        train_data['uid'] = train_data['uid'].map(raw_uid_to_new_uid_train_data).astype(int)
        train_data_for_train['uid'] = train_data_for_train['uid'].map(raw_uid_to_new_uid_train_data).astype(int) 
        raw_uid_to_new_uid_test_data = get_new_id(test_data)
        train_data_for_test['uid'] = train_data_for_test['uid'].map(raw_uid_to_new_uid_test_data).astype(int)
        test_data['uid'] = test_data['uid'].map(raw_uid_to_new_uid_test_data).astype(int)
        
        if True:
            data_tr = sparse.csr_matrix((np.ones_like(train_data_for_train['uid'].values),
                                     (train_data_for_train['uid'].values, train_data_for_train['sid'].values)), dtype='float64', shape=(train_data_for_train['uid'].max() + 1, self.n_items))
            data_te = sparse.csr_matrix((np.ones_like(train_data['uid'].values),
                                     (train_data['uid'].values, train_data['sid'].values)), dtype='float64', shape=(train_data['uid'].max() + 1, self.n_items))
            data_tr_for_te = sparse.csr_matrix((np.ones_like(train_data_for_test['uid'].values),
                                     (train_data_for_test['uid'].values, train_data_for_test['sid'].values)), dtype='float64', shape=(train_data_for_test['uid'].max() + 1, self.n_items))
            data_te_for_te = sparse.csr_matrix((np.ones_like(test_data['uid'].values),
                                     (test_data['uid'].values, test_data['sid'].values)), dtype='float64', shape=(test_data['uid'].max() + 1, self.n_items))
            return data_tr, data_te, data_tr_for_te, data_te_for_te
    
    def get_full_data(self):
        
        for index, value in enumerate(self.task_ids):
            item_of_recent_task = self.item_to_their_task_id[self.item_to_their_task_id['task_id'] == value]['new_id'].unique()
            test_data_task_value = self.test_data[self.test_data['sid'].isin(item_of_recent_task)].copy()
            train_data_task_value_for_test = self.train_data[self.train_data['uid'].isin(test_data_task_value['uid'].unique())].copy()
            train_data_task_value = self.train_data[self.train_data['sid'].isin(item_of_recent_task)].copy()
            train_data_task_value_for_train = self.train_data[self.train_data['uid'].isin(train_data_task_value['uid'].unique())].copy()
            train_data_task_value_for_train, train_data_task_value, train_data_task_value_for_test, test_data_task_value = self.load_tr_te_data(train_data_task_value_for_train, train_data_task_value, train_data_task_value_for_test, test_data_task_value)
            self.real_train_data[index] = [train_data_task_value_for_train.astype('float32'), train_data_task_value_for_test.astype('float32')]
            self.real_test_data[index] = [train_data_task_value.astype('float32'), test_data_task_value.astype('float32')]
            
            
    def load_tr_data_basic(self, train_data):
        def get_new_id(data):
            raw_uid_to_new_uid = {value: index for index, value in enumerate(data['uid'].unique())}
            return raw_uid_to_new_uid
        raw_uid_to_new_uid_train_data = get_new_id(train_data)
        train_data['uid'] = train_data['uid'].map(raw_uid_to_new_uid_train_data).astype(int)
        data_tr = sparse.csr_matrix((np.ones_like(train_data['uid'].values),
                                     (train_data['uid'].values, train_data['sid'].values)), dtype='float64', shape=(train_data['uid'].max() + 1, self.n_items))
        return data_tr
    def load_tr_te_data_basic(self, train_data_for_test, test_data):
        def get_new_id(data):
            raw_uid_to_new_uid = {value: index for index, value in enumerate(data['uid'].unique())}
            return raw_uid_to_new_uid
        raw_uid_to_new_uid_test_data = get_new_id(test_data)
        train_data_for_test['uid'] = train_data_for_test['uid'].map(raw_uid_to_new_uid_test_data).astype(int)
        test_data['uid'] = test_data['uid'].map(raw_uid_to_new_uid_test_data).astype(int)
        
        if True:
            data_tr_for_te = sparse.csr_matrix((np.ones_like(train_data_for_test['uid'].values),
                                     (train_data_for_test['uid'].values, train_data_for_test['sid'].values)), dtype='float64', shape=(train_data_for_test['uid'].max() + 1, self.n_items))
            data_te_for_te = sparse.csr_matrix((np.ones_like(test_data['uid'].values),
                                     (test_data['uid'].values, test_data['sid'].values)), dtype='float64', shape=(test_data['uid'].max() + 1, self.n_items))
            return data_tr_for_te, data_te_for_te
    def get_full_data_basic(self):
        for index, value in enumerate(self.task_ids):
            item_of_recent_task = self.item_to_their_task_id[self.item_to_their_task_id['task_id'] == value]['new_id'].unique()
            test_data_task_value = self.test_data[self.test_data['sid'].isin(item_of_recent_task)].copy()
            train_data_task_value_for_test = self.train_data[self.train_data['uid'].isin(test_data_task_value['uid'].unique())].copy()
            train_data_task_value_for_test, test_data_task_value = self.load_tr_te_data_basic(train_data_task_value_for_test, test_data_task_value)
            self.real_test_data[index] = [train_data_task_value_for_test.astype('float32'), test_data_task_value.astype('float32')]
        self.real_train_data = self.load_tr_data_basic(self.train_data.copy())
            
            