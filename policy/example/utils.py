import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import trimesh
import open3d as o3d
import matplotlib.pyplot as plt

import mujoco
import json
import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad





class ActionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir,mesh_path, seq_id,position_vector):
        super(ActionDataset).__init__()

        self.dataset_dir = dataset_dir
        self.seq_id = seq_id
        self.position_vector = position_vector
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
           
        seq_dir = os.path.join(self.dataset_dir, str(self.seq_id))
        robot_state_file = os.path.join(seq_dir, "robot_state.txt")
        mesh = trimesh.load(self.mesh_path)
        gt_vertices = mesh.vertices
        # Load the first line from robot_state.txt
        with open(robot_state_file, 'r') as f:
            lines = f.readlines()
            # empty file?

            # The example JSON is on one line, so parse the first line
        first_line_data = json.loads(lines[0])
            # "O_T_EE" is 16 floats representing a 4x4 transform
            # Typically, the translation is at indices [12,13,14] in row-major
        O_T_EE = first_line_data["O_T_EE"]
        q=first_line_data["q"]


        ee_position = [O_T_EE[12], O_T_EE[13], O_T_EE[14]]

        
        sample_id = self.seq_id
        ee_pose = torch.from_numpy(np.array(ee_position)).float()
        qpos = torch.from_numpy(np.array(q)).float()
        vertices = gt_vertices+self.position_vector
        vertices = torch.from_numpy(vertices).float()
        return sample_id,vertices,qpos, ee_pose





def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim



def load_data_from_teleop(dataset_dir,
                          xml_path,
                          mesh_path="/home/haozhe/Dropbox/physics/_data/allegro/wonik_allegro/assets/bluelego_convex.stl", 
                          dt=0.002):
    # in the dataset dir, there will be sequences of data range from 0 to 10
    # in each folder, there will be a robot_state.txt, load from it and you will 
    # the q pos of franka arm and end effector pose of franka arm

    # sample dataformart, there are lots of them in the txt, just load the first frame is fine
    # {"O_T_EE": [-0.750989,-0.609741,-0.253439,0,-0.600344,0.790312,-0.122452,0,0.274959,0.0601907,-0.95957,0,0.423825,-0.0351224,0.44687,1], "O_T_EE_d": [-0.750997,-0.609742,-0.253415,0,-0.60035,0.790311,-0.122428,0,0.274926,0.0601944,-0.959579,0,0.423824,-0.03512,0.446855,1], "F_T_NE": [0.707107,0.707107,0,0,-0.707107,0.707107,0,0,0,0,1,0,0,0,0.1034,1], "NE_T_EE": [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1], "F_T_EE": [0.707107,0.707107,0,0,-0.707107,0.707107,0,0,0,0,1,0,0,0,0.1034,1], "EE_T_K": [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1], "m_ee": 1.2, "F_x_Cee": [0,-0.06,0], "I_ee": [0.002,0,0,0,0.002,0,0,0,0.002], "m_load": 0, "F_x_Cload": [0,0,0], "I_load": [0,0,0,0,0,0,0,0,0], "m_total": 1.2, "F_x_Ctotal": [0,-0.06,0], "I_total": [0.002,0,0,0,0.002,0,0,0,0.002], "elbow": [-0.0454428,-1], "elbow_d": [-0.0454428,-1], "elbow_c": [0,0], "delbow_c": [0,0], "ddelbow_c": [0,0], "tau_J": [0.360375,-11.5484,0.237352,24.332,1.16504,3.11012,-0.0749632], "tau_J_d": [0,0,0,0,0,0,0], "dtau_J": [10.3877,15.0108,96.3704,3.08967,-5.67406,-15.6657,41.4151], "q": [-0.0618004,-0.594178,-0.0454428,-2.41595,0.0724627,2.09259,1.56008], "dq": [0.000303315,-0.000481621,-7.60882e-05,0.00081936,-0.000685416,0.000490441,-0.00110909], "q_d": [-0.0617958,-0.594162,-0.0454428,-2.41596,0.072462,2.09258,1.56009], "dq_d": [0,0,0,0,0,0,0], "ddq_d": [0,0,0,0,0,0,0], "joint_contact": [0,0,0,0,0,0,0], "cartesian_contact": [0,0,0,0,0,0], "joint_collision": [0,0,0,0,0,0,0], "cartesian_collision": [0,0,0,0,0,0], "tau_ext_hat_filtered": [0.338707,1.77291,0.00360254,-2.06713,0.374161,-0.559372,-0.0318697], "O_F_ext_hat_K": [0.532697,0.305178,-3.90308,0.448526,1.92044,0.32062], "K_F_ext_hat_K": [0.403062,0.399322,3.91012,-0.397217,-0.26771,-0.0407644], "O_dP_EE_d": [0,0,0,0,0,0], "O_ddP_O": [0,0,-9.81], "O_T_EE_c": [-0.750997,-0.609742,-0.253415,0,-0.60035,0.790311,-0.122428,0,0.274926,0.0601944,-0.959579,0,0.423824,-0.03512,0.446855,1], "O_dP_EE_c": [0,0,0,0,0,0], "O_ddP_EE_c": [0,0,0,0,0,0], "theta": [-0.0617747,-0.595009,-0.045425,-2.41412,0.0726004,2.09295,1.56007], "dtheta": [0,0,0,0,0,0,0], "current_errors": [], "last_motion_errors": [], "control_command_success_rate": 0, "robot_mode": "Idle", "time": 3451588}


    # the end effector position is the value we want to predict
    # the input is the vertices of the object  

    # also in the dataset_dir, there will be a object mesh, load the 6dof pose of the bounding box of this obj mesh
    # and the vertices

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    
    mj_model.opt.timestep = dt

    # For example, adjust actuator force ranges (if needed)
    mj_model.actuator_forcelimited[12]=True
    mj_model.actuator_forcelimited[14]=True
    mj_model.actuator_forcerange[14][0] = 0
    mj_model.actuator_forcerange[14][1] = 0.30
    mj_model.actuator_forcerange[12][0] = 0
    mj_model.actuator_forcerange[12][1] = 0.70

    mj_model.actuator_gear[14][0] = 0.3 
    mj_model.actuator_gear[14][1] = 0.3 
    mj_model.actuator_gear[14][2] = 0.3 

    mj_model.actuator_gear[12][0] = 0.3 
    mj_model.actuator_gear[12][1] = 0.3 
    mj_model.actuator_gear[12][2] = 0.3 

    body_name = "lego"
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    joint_id = mj_model.body_jntadr[body_id]
    qpos_addr = mj_model.jnt_qposadr[joint_id]
    num_dofs = 7  # free joint: 3 pos + 4 quat


    mesh = trimesh.load(mesh_path)

    vertices = mesh.vertices
    vertices = torch.from_numpy(vertices).float()

      # -----------------------------
    # 4) Load the object's 6/7 DoF pose (if you store it in a JSON)
    #    Then set the lego pose in mj_data.qpos
    # -----------------------------
    object_pose_file = os.path.join(dataset_dir, "object_pose.json")
    if os.path.isfile(object_pose_file):
        with open(object_pose_file, 'r') as f:
            object_pose = json.load(f)
        # Expecting { "position": [x, y, z], "orientation": [qw, qx, qy, qz] }
        pos = object_pose["position"]
        quat = object_pose["orientation"]
        mj_data.qpos[qpos_addr : qpos_addr + 3] = pos
        mj_data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat
    else:
        # If you don't have an object_pose.json, just leave it at default or raise a warning
        print(f"Warning: No object_pose.json found in {dataset_dir}; lego pose not set explicitly.")

    # -----------------------------
    # 5) Parse subdirectories [0..10], load first frame from robot_state.txt
    # -----------------------------


    # -----------------------------
    # 6) Return everything
    # -----------------------------
    return mj_model, mj_data

def load_data_handcontrol(dataset_dir, mesh_path,xml_path,batch_size_train, batch_size_val, default_pos):
    print(f'\nData from: {dataset_dir}\n')


    mj_model, mj_data = load_data_from_teleop(dataset_dir,xml_path ,mesh_path, dt=0.002)
    vector= default_pos[:3]

    # construct dataset and dataloader
    train_dataset = ActionDataset(dataset_dir,mesh_path,position_vector=vector)
    val_dataset = ActionDataset(dataset_dir,mesh_path,position_vector=vector)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader,mj_model, mj_data

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
