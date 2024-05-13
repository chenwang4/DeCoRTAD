#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIS dataset
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch
import yaml
from mld_diff_condition_rl import Diffusion
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import math

with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
trajs_state_tensor = torch.load('../Data/AIS/normal_states_ais.pt')
trajs_len = [len(traj) for traj in trajs_state_tensor]

sum([len(traj) for traj in trajs_state_tensor])

trajs_state_tensor = [traj for traj in trajs_state_tensor if len(traj)>=40]
step_size = config["AIS_Data"]["step_size"]
feat_size = config["AIS_Data"]["feat_size"]
device = config['Train']['device']
latent_dim = 32

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #print(f"Random seed set as {seed}")
set_seed(100)
def get_meanstd(trajectories, win_len=90, step_size=30):
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

def detour_generate(train_data, dev, num):
    trajs_detour = []
    for i in range(num):
        test_traj = train_data[i].numpy()
        alpha_s = random.randint(0, int(len(test_traj)*0.4)-1)
        alpha_e = alpha_s + int(len(test_traj)*0.6)
        traj_detour = list(test_traj[:alpha_s])

        delta_y = [0]
        while delta_y[-1] < dev:
            delta = delta_y[-1] + random.uniform(0.01, 0.08)
            delta_y.append(delta)


        y_sec = test_traj[alpha_s][1] + delta_y
        for i in range(len(y_sec)):
            traj_detour.append(np.array([test_traj[alpha_s][0], y_sec[i], 9.0, 0.0]))
            
        pre_lon = test_traj[alpha_s][1]
        for i in range(alpha_s, alpha_e+1):
            lat, lon, speed, course = test_traj[i]
            traj_detour.append(np.array([lat, y_sec[-1]+(lon-pre_lon), speed, course]))

        y_sec = traj_detour[-1][1] - delta_y
        for i in range(len(y_sec)):
            traj_detour.append(np.array([test_traj[alpha_e][0], y_sec[i], 9.0, 360.0]))
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
        if dist > 1.0:
            bridge_s = first_traj[len(first_traj)//2][:3]
            bridge_d = second_traj[len(second_traj)//2]
            lat_diff = bridge_d[0] - bridge_s[0]
            lon_diff = bridge_d[1] - bridge_s[1]
            angle = np.arctan(lon_diff/lat_diff)*180/np.pi
            delta_lat = [0]
            delta_lon = [0]
            if abs(lon_diff) > abs(lat_diff):
                while delta_lon[-1] < abs(lon_diff):
                    delta = delta_lon[-1] + random.uniform(0.01, 0.08)
                    delta_lon.append(delta)
                interval = abs(lat_diff) / len(delta_lon)
                delta_lat = [interval*i for i in range(len(delta_lon))] #positive
            else:
                while delta_lat[-1] < abs(lat_diff):
                    delta = delta_lat[-1] + random.uniform(0.01, 0.08)
                    delta_lat.append(delta)
                interval = abs(lon_diff) / len(delta_lat)
                delta_lon = [interval*i for i in range(len(delta_lat))]
            if 70 < len(delta_lat) < 25:
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
            
            for i in range(len(sec_lat)):
                bridge.append(np.array([sec_lat[i], sec_lon[i], 10.0, angle]))
                
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

indices = np.arange(len(trajs_state_tensor))

np.mean([len(i) for i in trajs_state_tensor])
X_train, X_test, indices_train, indices_test = train_test_split(trajs_state_tensor, indices, test_size=0.2, random_state=42)
state_mean, state_std = get_meanstd(X_train)
detour_anomalies = detour_generate(X_train, dev=1.0, num=500) 
rc_anomalies, _, _= route_changing_generate(train_data=X_train, num=500)
set_seed(100)
cur_win_len = 10
pre_win_len = 20
diff_model = Diffusion(feat_size=feat_size, current_win_len=cur_win_len, past_win_len=pre_win_len, t_range=60, latent_dim=latent_dim)
#diff_model = Diffusion(feat_size=feat_size, current_win_len=cur_win_len, past_win_len=pre_win_len, t_range=60, latent_dim=latent_dim)
#diff_model.load_state_dict(torch.load( './AIS/models_sensitivity/diff_model_temopralTransformerRL_slidingwindow_AIS_allNoise'+'_curwin'+str(cur_win_len)+'_prewin'+str(pre_win_len), map_location=torch.device('cpu')))
diff_model.eval()
diff_model.to('cpu')
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
kneigh_dist_test_normal_trajs = k_neigh_dist(latent_test_normal, sections_test_normal, neigh, n_neighbors)
kneigh_dist_test_anomaly_rc = k_neigh_dist(latent_test_rc, sections_test_anomaly, neigh, n_neighbors)
plt.figure()
plt.hist(kneigh_dist_test_normal_trajs, histtype='barstacked', rwidth=1.2, bins=50, label='normal')
plt.hist(kneigh_dist_test_anomaly_rc, bins=50, histtype='barstacked', rwidth=0.4,label='anomaly')
plt.xlabel('KNN distance (K=1)')
plt.ylabel('Number of trajectories')
plt.legend()
plt.show()
threshold = sorted(kneigh_dist_test_normal_trajs)[int(len(kneigh_dist_test_normal_trajs)*0.92)]
# algorithm = IsolationForest(contamination=0.06, random_state=33)
# algorithm.fit(np.array(kneigh_dist_test_normal_trajs).reshape(-1,1))
p_rec = []
r_rec = []
f1_rc = []
for i in range(10):
    p, r, f1 = detection_results(threshold, 900, kneigh_dist_test_normal_trajs, 100, kneigh_dist_test_anomaly_rc)
    p_rec.append(p)
    r_rec.append(r)
    f1_rc.append(f1)
print('--------route changing----------')
print('precision', np.mean(p_rec))
print('recall', np.mean(r_rec))
print('f1', np.mean(f1_rc), np.std(f1_rc))



test_x_anomaly, test_last_x_anomaly, sections_test_anomaly = slide_window(pre_win_len, cur_win_len, step_size, detour_anomalies)
latent_test_anomaly = get_latent(test_x_anomaly, test_last_x_anomaly, sections_test_anomaly, diff_model)
kneigh_dist_test_anomaly_detour = k_neigh_dist(latent_test_anomaly, sections_test_anomaly, neigh, n_neighbors)
print('----------detour--------')
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

