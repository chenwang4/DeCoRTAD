#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training for Chengdu Dataset
"""

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import yaml
from mld_diff_condition_rl import Diffusion
from sklearn.model_selection import train_test_split
import math
from math import sin, cos, sqrt, atan2, radians
from sklearn.cluster import MeanShift
with open('./config_chengdu.yaml', 'r') as file:
    config = yaml.safe_load(file)

userdata = '../Data/Chengdu/gps_20161101'
step_size = config["Chengdu_Data"]["step_size"]
feat_size = 2
device = config['Train']['device']
latent_dim = 16
pre_win_len = 106
cur_win_len = 50

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #print(f"Random seed set as {seed}")
def get_meanstd(trajectories, win_len=120, step_size=30):
    x_feat = []
    sections = []
    for traj in trajectories:
        traj_len = len(traj)
        num_section = int((traj_len - win_len)/step_size + 1)
        if num_section < 1:
            continue
        f = []
        for k in range(num_section):
            w = np.array(traj[step_size*k:win_len + step_size*k]).reshape(-1)
            f.append(w)
        x_feat += f
        sections.append(num_section)
    state_mean = np.mean(np.array(x_feat).astype(float).reshape(-1, feat_size), axis=0)
    state_std = np.std(np.array(x_feat).astype(float).reshape(-1,feat_size), axis=0) + 1e-6
    return state_mean, state_std

def get_batch(data, batch_size, pre_win_len, cur_win_len, device):
        batch_inds = np.random.choice(
            np.arange(len(data)),
            size=batch_size,
        )
        #s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        s, past_s = [], []
        for i in range(batch_size):
            traj_s_batch = data[int(batch_inds[i])]
            #traj_a_batch = y_train[int(batch_inds[i])]
            si = random.randint(pre_win_len, traj_s_batch.shape[0] - 1)

            # get sequences from dataset
            s.append(traj_s_batch[si:si + cur_win_len].reshape(1, -1, feat_size))
            past_s.append(traj_s_batch[si-pre_win_len:si].reshape(1, -1, feat_size))
            
        
            # if sequence length is smaller than the window_size(win_len), then padding 
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, cur_win_len - tlen, feat_size)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            
            past_s[-1] = (past_s[-1] - state_mean) / state_std
            
            
        s = torch.from_numpy(np.concatenate(s, axis=0).astype(float)).to(dtype=torch.float32, device=device)
        past_s = torch.from_numpy(np.concatenate(past_s, axis=0).astype(float)).to(dtype=torch.float32, device=device)
        
        return s, past_s

def slide_window(pre_win_len, cur_win_len, step_size, trajectories):
    x_feat, last_x_feat, sections = [], [], []
    sections = []
    for traj in trajectories:
        traj_len = len(traj)
        num_section = int((traj_len - pre_win_len - cur_win_len)/step_size + 1)
        if num_section < 1:
            continue
        f, last_f = [], []
        for k in range(num_section):
            w = np.array(traj[pre_win_len+step_size*k:pre_win_len + cur_win_len + step_size*k]).reshape(-1)
            f.append(w)
            last_w = np.array(traj[step_size*k:pre_win_len + step_size*k]).reshape(-1)
            last_f.append(last_w)
        x_feat += f
        last_x_feat += last_f
        sections.append(num_section)
    return np.array(x_feat), np.array(last_x_feat), sections

def latent_features(data_x, data_last_x, diff_model):
    data_x = ((np.array(data_x).reshape(-1, feat_size) - state_mean) / state_std).reshape(-1, feat_size*cur_win_len)
    data_last_x = ((np.array(data_last_x).reshape(-1, feat_size) - state_mean) / state_std).reshape(-1, feat_size*pre_win_len)
    batch_cur = torch.from_numpy(data_x.astype(float)).to(dtype=torch.float32, device='cpu').reshape(-1,cur_win_len,feat_size) 
    batch_last = torch.from_numpy(data_last_x.astype(float)).to(dtype=torch.float32, device='cpu').reshape(-1,pre_win_len,feat_size)
    # print(batch_cur.shape)
    # print(batch_last.shape)
    latent_past_sample = diff_model.autoencoder_past(batch_last.reshape(batch_last.shape[0],-1)).unsqueeze(1) 
    batch_cond = torch.cat((batch_cur, latent_past_sample.reshape(batch_cur.shape[0], -1, feat_size)), axis=1)   
    latent_clean_sample = diff_model.autoencoder(batch_cond).unsqueeze(1)  
    return latent_clean_sample.squeeze().to('cpu').detach().numpy()

def sec_to_traj(dist, sections):
    '''
    convert dist array from per section to per trajectory
    '''
    dist_trajs = []
    current_index = 0
    for i in range(len(sections)):
        len_traj = sections[i]
        dist_trajs.append(max(dist[current_index:len_traj+current_index, -1]))
        current_index += len_traj
    return dist_trajs

def k_neigh_dist(latent_trajs, sections, knn_model, n_neighbors):
    dist, _ = knn_model.kneighbors(latent_trajs, n_neighbors, return_distance=True)
    dist_trajs = sec_to_traj(dist, sections)
    return dist_trajs


def detection_results(thres, num_norm, kneigh_dist_test_normal_trajs, num_anom, kneigh_dist_test_anomaly_trajs):
    test_data = np.concatenate((random.sample(kneigh_dist_test_normal_trajs, num_norm), random.sample(kneigh_dist_test_anomaly_trajs, num_anom)), axis=0).reshape(-1,1)
    #pred_test = algorithm.predict(test_data)    
    pred_test = np.array([-1 if i > thres else 1 for i in test_data])
    tp = sum(pred_test[-num_anom:]==-1)
    fp = sum(pred_test[:-num_anom]==-1)   
    fn = num_anom - tp
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    f1 = 2*p*r/(p+r)
    return p, r, f1

def diff_model_train(cur_win_len, pre_win_len):
    diff_model = Diffusion(feat_size=feat_size, current_win_len=cur_win_len, past_win_len=pre_win_len, t_range=60, latent_dim=latent_dim)
    diff_model.to(config["Train"]["device"])
    optim = torch.optim.Adam(diff_model.parameters(), lr=config["Train"]["lrate"]) #config["Train"]["lrate"]
    for ep in range(100):
        results_ep = [ep]
        diff_model.train()
        loss_ep, n_batch = 0, 0
        for i in range(100):
            batch = get_batch(X_train, config["Train"]["batch_size"], pre_win_len, cur_win_len, device)
            loss = diff_model.get_loss(batch, "AllLevel")
            optim.zero_grad()
            loss.backward()
            loss_ep += loss.detach().item()
            n_batch += 1
            optim.step()
        print(f"train loss: {loss_ep/n_batch:.4f}")
        results_ep.append(loss_ep / n_batch)
    diff_model.eval()
    return diff_model

def detour_generate(X_train, dev, num):
    trajs_detour = []
    for i in range(num):
        test_traj = X_train[i]
        alpha_s = random.randint(0, int(len(test_traj)*0.4)-1)
        alpha_e = alpha_s + int(len(test_traj)*0.6)
        traj_detour = list(test_traj[:alpha_s])

        delta_y = [0]
        while delta_y[-1] < dev:
            delta = delta_y[-1] + random.uniform(0.0001, 0.0003)
            delta_y.append(delta)


        y_sec = test_traj[alpha_s][1] + np.array(delta_y)
        #t_sec = test_traj[alpha_s][2] + np.arange(3, 3*(len(y_sec)+1), 3)
        for i in range(len(y_sec)):
            traj_detour.append(np.array([test_traj[alpha_s][0], y_sec[i]]))
        
        
        pre_lon = test_traj[alpha_s][1]
        #t2_sec = t_sec[-1] + np.arange(3, 3*(alpha_e-alpha_s+2), 3)
        for i in range(alpha_s, alpha_e+1):
            lat, lon = test_traj[i]
            traj_detour.append(np.array([lat, y_sec[-1]+(lon-pre_lon)]))

        y_sec = traj_detour[-1][1] - delta_y
        #t3_sec = t2_sec[-1] + np.arange(3, 3*(len(y_sec)+1), 3)
        for i in range(len(y_sec)):
            traj_detour.append(np.array([test_traj[alpha_e][0], y_sec[i]]))
        traj_detour += list(test_traj[alpha_e:])
        trajs_detour.append(traj_detour)
    return trajs_detour


def route_changing_generate(train_data, num):
    trajs_rc = []
    num_trajs = 0
    while num_trajs < num:
        first_traj = random.sample(train_data, 1)[0]
        second_traj = random.sample(train_data, 1)[0]
        second_half = second_traj[len(second_traj)//2:]
        first_dest = first_traj[-1][:2]
        second_dest = second_traj[-1][:2]
        dist = math.dist(first_dest, second_dest) #get distance
        if dist > 0.1:
            bridge_s = first_traj[len(first_traj)//2][:3]
            bridge_d = second_traj[len(second_traj)//2]
            lat_diff = bridge_d[0] - bridge_s[0]
            lon_diff = bridge_d[1] - bridge_s[1]
            delta_lat = [0]
            delta_lon = [0]
            if abs(lon_diff) > abs(lat_diff):
                while delta_lon[-1] < abs(lon_diff):
                    delta = delta_lon[-1] + random.uniform(1e-5, 3e-4)
                    delta_lon.append(delta)
                interval = abs(lat_diff) / len(delta_lon)
                delta_lat = [interval*i for i in range(len(delta_lon))] #positive
            else:
                while delta_lat[-1] < abs(lat_diff):
                    delta = delta_lat[-1] + random.uniform(1e-5, 3e-4)
                    delta_lat.append(delta)
                interval = abs(lon_diff) / len(delta_lat)
                delta_lon = [interval*i for i in range(len(delta_lat))]
            if len(delta_lat) < 30:
                continue
            #construct the bridge
            bridge = []
            if lat_diff > 0 and lon_diff > 0:
                sec_lat = bridge_s[0] + np.array(delta_lat)
                sec_lon = bridge_s[1] + np.array(delta_lon)
            elif lat_diff > 0 and lon_diff < 0:
                sec_lat = bridge_s[0] + np.array(delta_lat)
                sec_lon = bridge_s[1] - np.array(delta_lon)
            elif lat_diff < 0 and lon_diff > 0:
                sec_lat = bridge_s[0] - np.array(delta_lat)
                sec_lon = bridge_s[1] + np.array(delta_lon)
            elif lat_diff < 0 and lon_diff < 0:
                sec_lat = bridge_s[0] - np.array(delta_lat)
                sec_lon = bridge_s[1] - np.array(delta_lon)
            else:
                print('error')
                continue
            #time_change = bridge_s[-1]
            for i in range(len(sec_lat)):
                bridge.append(np.array([sec_lat[i], sec_lon[i]]))
                #time_change += 3
            # for i in range(len(second_half)):
            #     second_half[i][2] = time_change
            #     time_change += 3
            # connect three parts
            traj_rc = list(first_traj[:len(first_traj)//2]) + bridge + list(second_half)
            trajs_rc.append(traj_rc)
        else:
            continue
        num_trajs = len(trajs_rc)
    return trajs_rc, first_traj, second_traj


def get_latent(x, last_x, sections, model):
    latent_samples = []
    count = 0
    for i in range(len(sections)//100): #len(sections)
        len_traj = sum(sections[i*100:(i+1)*100]) #sum(sections[i:i+100])
        latent = latent_features(x[count:len_traj+count], last_x[count:len_traj+count], model)
        latent_samples.append(latent)
        count += len_traj
    latent_samples = np.array([sample for latent_sample in latent_samples for sample in latent_sample]).reshape(-1, latent_dim)
    return latent_samples

def get_distance(s,d):
    R = 6373.0
    lat1 = radians(s[0])
    lon1 = radians(s[1])
    lat2 = radians(d[0])
    lon2 = radians(d[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def Chengdu_data_preprocessing(data_address):
    names = ['driver_id', 'order_id', 'time', 'longi', 'lati']
    df = pd.read_table(data_address, header = None, names=names, sep=',')
    data = np.c_[df['lati'].values, df['longi'].values, df['order_id'].values]
    trajs = []
    s_ind = 0
    d_ind = 1  
    for i in range(len(data)-1):
        if data[i][-1] == data[i+1][-1]:
            d_ind += 1
        else:
            traj = data[s_ind:d_ind, :2]
            if 2000 > len(traj) > 260:
                # add information of drop-off location
                #traj = np.concatenate((traj, np.tile(traj[-1, 0:2], [len(traj),1])), axis=1)
                trajs.append(traj)
            s_ind = i + 1
            d_ind += 1  

    trajs_longer = []
    dests = []
    dists = []
    for t in trajs:
        dist = get_distance(t[0][0:2],t[-1][0:2])    
        if dist > 3:
            trajs_longer.append(t)
            dists.append(dist)
            dests.append(list(t[-1][0:2]))
    # clustering based on destinations
    clustering = MeanShift(bandwidth=0.004).fit(dests)
    labels = clustering.labels_
    n_clusters_ = len(set(labels))
    trajs_allcluster = []
    for label in range(n_clusters_):
        trajs_ind = np.where(labels==label)[0]   
        if len(trajs_ind) > 100:
            traj_cluster = [trajs_longer[i] for i in trajs_ind]
            trajs_allcluster.append(traj_cluster)

    n_clusters = 6
    trajs_n_clusters = [trajs_allcluster[0]]
    i = 2
    while len(trajs_n_clusters) < n_clusters:
        flag = 0
        picked_dests = [trajs_n_clusters[n][0][-1][0:2] for n in range(len(trajs_n_clusters))]
        cluster = trajs_allcluster[i]
        cluster_dest = cluster[0][-1][0:2]
        i += 1
        for picked_dest in picked_dests:
            if get_distance(cluster_dest, picked_dest) < 3:
                flag = 1
                break
        if flag == 0:
            trajs_n_clusters.append(cluster)
        
    trajs_chengdu = [traj for cluster in trajs_n_clusters for traj in cluster]
    return trajs_chengdu

trajs_chengdu = Chengdu_data_preprocessing(userdata)
#np.mean([len(i) for i in trajs_chengdu])
indices = np.arange(len(trajs_chengdu))
X_train, X_test, indices_train, indices_test = train_test_split(trajs_chengdu, indices, test_size=0.2, random_state=42)
state_mean, state_std = get_meanstd(X_train)
detour_anomalies = detour_generate(X_train, dev=0.04, num=500) 
rc_anomalies, _, _= route_changing_generate(train_data=X_train, num=500)
set_seed(103)
print('--------------------------')
print('cur_win_len', cur_win_len)
print('pre_win_len', pre_win_len)
diff_model = diff_model_train(cur_win_len, pre_win_len)
diff_model.eval()
train_x_normal, train_last_x_normal, sections_train_normal = slide_window(pre_win_len, cur_win_len, step_size, X_train)
test_x_normal, test_last_x_normal, sections_test_normal = slide_window(pre_win_len, cur_win_len, step_size, X_test)
test_x_anomaly, test_last_x_anomaly, sections_test_anomaly = slide_window(pre_win_len, cur_win_len, step_size, rc_anomalies)
latent_clean_sample = get_latent(train_x_normal, train_last_x_normal, sections_train_normal , diff_model)
latent_test_normal = get_latent(test_x_normal, test_last_x_normal, sections_test_normal, diff_model)
latent_test_rc = get_latent(test_x_anomaly, test_last_x_anomaly, sections_test_anomaly, diff_model)
n_neighbors = 1
neigh = NearestNeighbors(n_neighbors=n_neighbors)
neigh.fit(latent_clean_sample)
#kneigh_dist_train_trajs = k_neigh_dist(latent_clean_sample, sections_train_normal, neigh, n_neighbors)
kneigh_dist_test_normal_trajs = k_neigh_dist(latent_test_normal, sections_test_normal[:1000], neigh, n_neighbors)
kneigh_dist_test_anomaly_rc = k_neigh_dist(latent_test_rc, sections_test_anomaly, neigh, n_neighbors)
plt.figure()
plt.hist(kneigh_dist_test_normal_trajs, histtype='barstacked', rwidth=1.2, bins=50, label='normal')
plt.hist(kneigh_dist_test_anomaly_rc, bins=50, histtype='barstacked', rwidth=0.4,label='anomaly')
plt.xlabel('KNN distance (K=1)')
plt.ylabel('Number of trajectories')
plt.legend()
plt.show()

threshold = sorted(kneigh_dist_test_normal_trajs)[int(len(kneigh_dist_test_normal_trajs)*0.95)]
p_rec = []
r_rec = []
f1_rc = []
for i in range(10):
    p, r, f1 = detection_results(threshold, 900, kneigh_dist_test_normal_trajs, 100, kneigh_dist_test_anomaly_rc)
    p_rec.append(p)
    r_rec.append(r)
    f1_rc.append(f1)
print('--------route chaging----------')
print('precision', np.mean(p_rec))
print('recall', np.mean(r_rec))
print('f1', np.mean(f1_rc), np.std(f1_rc))


print('----------detour--------')
test_x_anomaly, test_last_x_anomaly, sections_test_anomaly = slide_window(pre_win_len, cur_win_len, step_size, detour_anomalies)
latent_test_anomaly = get_latent(test_x_anomaly, test_last_x_anomaly, sections_test_anomaly, diff_model)
kneigh_dist_test_anomaly_detour = k_neigh_dist(latent_test_anomaly, sections_test_anomaly, neigh, n_neighbors)
p_rec = []
r_rec = []
f1_detour = []
for i in range(10):
    p, r, f1 = detection_results(threshold, 900, kneigh_dist_test_normal_trajs, 100, kneigh_dist_test_anomaly_detour)
    p_rec.append(p)
    r_rec.append(r)
    f1_detour.append(f1)
print('precision', np.mean(p_rec))
print('recall', np.mean(r_rec))
print('f1', np.mean(f1_detour), np.std(f1_detour))
