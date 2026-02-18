import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import mujoco as mj
from torch.utils.data import Dataset, DataLoader

import open3d as o3d



import mujoco.viewer as viewer
import open3d as o3d
import time
import sys



# add contact and grasp validation filter for the input data:


# under clipping, self collision and active contact with object




# so i will try to find the lowest hand position that able to 



def validate_action_dataset_with_gt(gt_action_data, threshold=0.5):
    """
    Loads a numpy array from `gt_action_data` (shape [N, 16] assumed).
    For each row (action) in that array, we compare it to the three 
    "in_range_data" references. If the L2 distance (or any chosen metric) 
    to any reference is below `threshold`, we say it's valid; else not good.

    Returns the number of valid actions.
    """
    # gt pose 1
    gtpose1 = np.array([
        3.24,4.48,3.17,4.12,
        3.23,4.53,3.22,4.05,
        3.21,4.52,3.22,4.13,
        4.75,3.25,3.28,3.63
    ])

    # gt pose 2
    gtpose2 = np.array([
        3.25,4.28,3.73,3.49,
        3.25,4.43,3.76,3.23,
        4.27,4.45,4.07,3.24,
        4.86,3.25,3.23,3.49
    ])

    # gt pose 3
    gtpose3 = np.array([
        3.27,3.81,4.84,3.22,
        3.07,4.51,4.12,3.24,
        3.17,5.04,3.24,3.71,
        4.32,3.26,3.15,3.63
    ])

    # Subtract 3.25 to get in_range versions
    in_range_data_1 = gtpose1 - 3.25
    in_range_data_2 = gtpose2 - 3.25
    in_range_data_3 = gtpose3 - 3.25

    # Load user actions
    action = np.load(gt_action_data)  # shape [N,16] expected

    valid_count = 0
    total = len(action)

    for i in range(total):
        # Compute distances to each reference pose
        dist1 = np.linalg.norm(action[i] - in_range_data_1)
        dist2 = np.linalg.norm(action[i] - in_range_data_2)
        dist3 = np.linalg.norm(action[i] - in_range_data_3)

        # If any distance is below threshold, consider it valid
        if (dist1 < threshold) or (dist2 < threshold) or (dist3 < threshold):
            print(f"Action[{i}] is valid. dist1={dist1:.3f}, dist2={dist2:.3f}, dist3={dist3:.3f}")
            valid_count += 1
        else:
            print(f"Action[{i}] is NOT good. dist1={dist1:.3f}, dist2={dist2:.3f}, dist3={dist3:.3f}")

    print(f"Valid actions: {valid_count}/{total}")
    return valid_count



def positional_encoding_3d_torch(xyz: torch.Tensor, num_freqs: int = 4) -> torch.Tensor:
    """
    3D positional encoding for each (x, y, z) using PyTorch.

    For each coordinate in xyz, we generate:
      sin(2^k * coord), cos(2^k * coord)
    for k in [0 .. num_freqs-1].

    If 'num_freqs' is 4, we get:
      2 * 4 = 8 values per coordinate
      => 8 * 3 = 24 values total for (x,y,z) per point.

    Args:
        xyz (torch.Tensor): A tensor of shape (N, 3), 
                            where each row is (x, y, z).
        num_freqs (int): Number of frequency bands.

    Returns:
        torch.Tensor: Shape (N, 3 * 2 * num_freqs).
                      Example: If N=8 and num_freqs=4, 
                      the result is shape (8, 24).
    """
    assert xyz.dim() == 2 and xyz.shape[1] == 3, "xyz must be (N, 3)"
    
    N = xyz.shape[0]
    out_dim = 3 * 2 * num_freqs
    # Create the output tensor on the same device as xyz
    enc = torch.zeros(N, out_dim, dtype=xyz.dtype, device=xyz.device)
    
    idx_offset = 0
    for coord_i in range(3):
        coord_vals = xyz[:, coord_i]  # shape (N,)
        for freq_i in range(num_freqs):
            freq = 2.0 ** freq_i
            sin_col = torch.sin(freq * coord_vals)
            cos_col = torch.cos(freq * coord_vals)
            enc[:, idx_offset] = sin_col
            enc[:, idx_offset + 1] = cos_col
            idx_offset += 2

    return enc


###########################################################
# 1) MUJOCO ENVIRONMENT
###########################################################
class HandEnv:
    """
    A minimal MuJoCo environment for a multi-finger hand.
    Uses `_compute_reward_4d` to get a 4D reward vector 
    (example code for demonstration).
    """

    def __init__(self, 
                 model_xml_path,
                 action_dim=16,
                 frame_skip=5,       # how many mj steps we do per 'env step'
                 episode_length=200  # max steps per episode
                 ):
        """
        :param model_xml_path: path to the MuJoCo XML (e.g., scene_apple.xml)
        :param action_dim: dimension of the hand's action space
        :param frame_skip: number of physics steps per env step
        :param episode_length: maximum number of env steps before 'done'
        """
        self.model_xml_path = model_xml_path
        self.action_dim = action_dim
        self.frame_skip = frame_skip
        self.episode_length = episode_length
        
        # Load the MuJoCo model
        self.mj_model = mj.MjModel.from_xml_path(model_xml_path)
        self.mj_data = mj.MjData(self.mj_model)

        # Example: define your hand joint names (must match your XML)
        self.hand_joint_names = [
            "ffj0", "ffj1", "ffj2", "ffj3",
            "mfj0", "mfj1", "mfj2", "mfj3",
            "rfj0", "rfj1", "rfj2", "rfj3",
            "thj0", "thj1", "thj2", "thj3",
        ]
        if len(self.hand_joint_names) != self.action_dim:
            raise ValueError("action_dim does not match the number of joints")

        # Episode tracking
        self.current_step = 0
        self.done = False

        # (Optional) define observation_dim.
        # For demonstration, we use a naive "observation" = all QPOS and QVEL (hand only).
        self.obs_dim = 2 * len(self.hand_joint_names)

        # If you want random initialization, etc., define an RNG
        self.rng = np.random.default_rng(12345)

    def reset(self):
        """
        Resets the environment state. 
        Returns the initial observation.
        """
        self.current_step = 0
        self.done = False

        # Clear simulation state
        mj.mj_resetData(self.mj_model, self.mj_data)

        # Example: set all hand joints to 0.0
        for jname in self.hand_joint_names:
            joint_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_JOINT, jname)
            qpos_adr = self.mj_model.jnt_qposadr[joint_id]
            self.mj_data.qpos[qpos_adr] = 0.0
            # also zero velocity
            qvel_adr = self.mj_model.jnt_dofadr[joint_id]
            self.mj_data.qvel[qvel_adr] = 0.0

        # Forward to compute positions, contacts, etc.
        mj.mj_forward(self.mj_model, self.mj_data)

        return self._get_obs()

    def step(self, action):
        """
        Apply 'action' to the environment, step the physics.
        """
        action = np.clip(action, -1.0, 1.0)
        if len(action) != self.mj_model.nu:
            raise ValueError(f"Action length {len(action)} != model.nu ({self.mj_model.nu})")

        # Step the simulation with 'frame_skip'
        for _ in range(self.frame_skip):
            self.mj_data.ctrl[:] = action
            mj.mj_step(self.mj_model, self.mj_data)

        self.current_step += 1
        if self.current_step >= self.episode_length:
            self.done = True

        return self._get_obs(), self.done

    def _get_obs(self):
        """
        A naive observation: [hand joint angles, hand joint velocities].
        """
        qpos_list = []
        qvel_list = []

        for jname in self.hand_joint_names:
            joint_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_JOINT, jname)
            qpos_adr = self.mj_model.jnt_qposadr[joint_id]
            qvel_adr = self.mj_model.jnt_dofadr[joint_id]

            qpos_list.append(self.mj_data.qpos[qpos_adr])
            qvel_list.append(self.mj_data.qvel[qvel_adr])

        obs = np.concatenate([qpos_list, qvel_list], axis=0)  # shape [2 * action_dim]
        return obs


    # geoms you want to ignore when deciding “special”
   

    def compute_4d_reward(self):
        """
        Returns a 4-channel reward vector:
        r0 – any contact with the table (“plate” in either geom name)
        r1 – any contact that involves the object “U”
        r2 – at least 3 simultaneous contacts
        r3 – **special**: at least one contact whose two geom IDs are
            both *not* in {0, 1, 43, 44}
        """
        _EXCLUDE_GEOMS = {0, 1, 44}
        table_contact   = 0.0
        object_contact  = 0.0
        many_contacts   = 0.0
        special_contact = 0.0

        contact_count = self.mj_data.ncon

        for i in range(contact_count):
            c   = self.mj_data.contact[i]
            g1, g2 = c.geom1, c.geom2

            # decode names once (helps debugging / other rules)
            n1 = mj.mj_id2name(self.mj_model, mj.mjtObj.mjOBJ_GEOM, g1)
            n2 = mj.mj_id2name(self.mj_model, mj.mjtObj.mjOBJ_GEOM, g2)

            # r0: table
            # if "plate" in n1 or "plate" in n2:
            #     table_contact = 1.0

            # # r1: object “U”
            # if "U" in n1 or "U" in n2:
            #     object_contact = 1.0

            # r3: special – both geoms outside the exclusion set
            if g1 not in _EXCLUDE_GEOMS and g2 not in _EXCLUDE_GEOMS:
                special_contact = 1.0

        # r2: many contacts (>=3)
        if contact_count >= 3:
            many_contacts = 1.0

        reward=np.array(
            [many_contacts, special_contact],
            dtype=np.float32,
        )

        # print(f"Reward: {reward}")

        # # normalize reward to make it between 0 and 1
        # reward = (reward - np.min(reward)) / (np.max(reward) - np.min(reward))
        return reward

    def run_with_renderer(self, control_signal, steps=10000, camera_name="side", width=640, height=480):
            """
            Example method that:
            1) Resets the env
            2) Opens a MuJoCo passive viewer for visualization and event handling
            3) Steps 'steps' times using 'control_signal' as the action
            4) Renders each frame (offscreen) from the specified camera
            5) Returns a list of frames (each frame is a NumPy RGBA array)
            6) Allows stopping early by pressing the ESC key in the viewer
            """
            # List to store frames
            frames = []

            # Reset the environment
            # self.reset()

            # Create an off-screen mj.Renderer for capturing frames
            offscreen_renderer = mj.Renderer(self.mj_model, width=width, height=height)

            # Launch the passive viewer
            with viewer.launch_passive(self.mj_model, self.mj_data) as viewer1:

                step_i = 0
                self.reset()
                while viewer1.is_running() and step_i < steps:
                    # Check if there's a new event in the viewer

                    # If the viewer is closed or not running, break
                    if not viewer1.is_running():
                        break

                    # Use the provided control signal for action
                    action = control_signal

                    
                    # Step the environment (your env logic)
                    obs, done = self.step(action)

                    # self.reset()
                    # Forward the state in MuJoCo (important for updating the simulation)
                    with viewer1.lock():
                        mj.mj_forward(self.mj_model, self.mj_data)

                    # Synchronize the viewer to display the latest state
                    viewer1.sync()

                    # Now capture the frame via the offscreen renderer
                    offscreen_renderer.update_scene(self.mj_data, camera=camera_name)
                    pixels = offscreen_renderer.render()
                    frames.append(pixels)

                    # if done:
                    #     print("Done condition reached.")
                    #     break

                    step_i += 1

                    # Optionally, small delay to slow things down visually
                    time.sleep(0.01)

            return frames


###########################################################
# 2) MLP MODEL (20D OUTPUT: 16 ACTION + 4 REWARD)
###########################################################

class GraspMLP(nn.Module):
    """
    Outputs 16D for joint actions, 4D for reward. 
    The reward head has a Sigmoid so the final output 
    is guaranteed in [0,1].
    """
    def __init__(self, input_dim=24, hidden_dim=256, action_dim=16, reward_dim=4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Separate heads:
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, reward_dim),
            nn.Sigmoid()  # ensures [0,1]
        )

    def forward(self, x):
        """
        x: (batch_size, input_dim)
        returns:
          pred_actions:  (batch_size, 16)  (no activation)
          pred_reward:   (batch_size, 4)   in [0,1]
        """
        feat = self.shared(x)               # (B, hidden_dim)
        pred_actions = self.action_head(feat).mean(dim=1)  # [B,16]
        pred_reward  = self.reward_head(feat).mean(dim=1)  # Sigmoid => [0,1]
        return pred_actions, pred_reward


###########################################################
# 3) DATASET + DATALOADER (PHASE 1)
###########################################################

class Phase1Dataset(Dataset):
    """
    For Phase 1, we have:
      - input vertices (e.g. 192-d) 
      - ground-truth actions (16-d)
      - fixed reward = [1,1,1,1]
    """
    def __init__(self, gt_mesh_path, gt_control_path):
        super().__init__()
        self.gt_mesh_path = gt_mesh_path
        self.gt_control_path = gt_control_path

        # Read mesh and convert vertices
        mesh = o3d.io.read_triangle_mesh(self.gt_mesh_path)
        input_vertices_np = np.asarray(mesh.vertices)               # shape ~ [V, 3]
        input_vertices = torch.tensor(input_vertices_np,
                                      dtype=torch.float32,
                                      device="cuda")

        # Apply some encoding (or keep it as-is)
        self.vertices = positional_encoding_3d_torch(input_vertices) # shape [V, ?]

        # Load ground-truth actions
        # shape [N, 16] expected
        gt_actions_np = np.load(self.gt_control_path)
        self.gt_actions = torch.tensor(gt_actions_np,
                                       dtype=torch.float32,
                                       device="cuda")

        # --------------------- CLIP STEP ---------------------
        sim_min, sim_max = LEAPsim_limits()
        tmin = torch.tensor(sim_min, dtype=torch.float32, device="cuda")
        tmax = torch.tensor(sim_max, dtype=torch.float32, device="cuda")
        # Broadcast clamp across all rows [N,16]
        self.gt_actions = torch.clamp(self.gt_actions, min=tmin, max=tmax)
        # -----------------------------------------------------
        count = validate_action_dataset_with_gt(self.gt_control_path)
        # Reward is fixed to [1,1,1,1]
        self.fixed_reward = torch.ones(2, device="cuda")

        # We assume the dataset size = number of GT actions
        # (One input for each action)
        self.num_samples = self.gt_actions.shape[0]


    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # For Phase 1, we feed the same 'input' (mesh features)
        # but pull out a different gt action (and reward) each time.
        # Or if your input is truly the same for all samples, you
        # might only store it once.  
        sample_input  = self.vertices      # [?, ?] or [192]
        sample_action = self.gt_actions[idx]      # [16]
        sample_reward = self.fixed_reward         # [4]
        return {
            "input": sample_input,
            "gt_action": sample_action,
            "gt_reward": sample_reward,
        }
    


def train_phase_1(model, dataloader, optimizer, epochs=100, device="cuda"):
    """
    Phase 1:
      - pred_actions ~ gt_actions (MSE)
      - pred_reward ~ [1,1,1,1]   (BCE, since model outputs sigmoid)
    """
    mse = nn.MSELoss()
    bce = nn.BCELoss()  # we have Sigmoid in the model => BCE
    model.train()
    model.to(device)

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            # input_data shape: [B, 1, 24], we might want to flatten
            input_data = batch["input"].to(device)  # [B, 1, 24]
            input_data = input_data.squeeze(1)      # [B, 24]
            gt_action  = batch["gt_action"].to(device)  # [B, 16]
            gt_reward  = batch["gt_reward"].to(device)  # [B, 4]

            optimizer.zero_grad()
            pred_action, pred_reward = model(input_data)  
            # shapes: [B,16], [B,4], each in [0,1] for reward

            # 1) action loss
            loss_action = mse(pred_action, gt_action)

            # 2) reward loss (we want it to match [1,1,1,1])
            loss_reward = bce(pred_reward, gt_reward)

            loss = loss_action +loss_reward*0.7
            loss.backward()
            optimizer.step()

            bs = input_data.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

        avg_loss = total_loss / total_samples
        if epoch % 10 == 0:
            print(f"[Phase 1] Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")



def train_phase_3(model, dataloader, optimizer, epochs=100, device="cuda"):
    """
    Phase 1:
      - pred_actions ~ gt_actions (MSE)
      - pred_reward ~ [1,1,1,1]   (BCE, since model outputs sigmoid)
    """
    mse = nn.MSELoss()
    bce = nn.BCELoss()  # we have Sigmoid in the model => BCE
    model.train()
    model.to(device)

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            # input_data shape: [B, 1, 24], we might want to flatten
            input_data = batch["input"].to(device)  # [B, 1, 24]
            input_data = input_data.squeeze(1)      # [B, 24]
            gt_action  = batch["gt_action"].to(device)  # [B, 16]
            gt_reward  = batch["gt_reward"].to(device)  # [B, 4]

            optimizer.zero_grad()
            pred_action, pred_reward = model(input_data)  
            # shapes: [B,16], [B,4], each in [0,1] for reward

            # 1) action loss
            loss_action = mse(pred_action, gt_action)



            loss = loss_action 
            loss.backward()
            optimizer.step()

            bs = input_data.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

        avg_loss = total_loss / total_samples
        if epoch % 10 == 0:
            print(f"[Phase 1] Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")



def train_phase_2(model, dataloader, env, optimizer, epochs=100, device="cuda"):
    """
    Phase 2:
      1) Model predicts [16 actions, 4 reward].
      2) We take predicted actions (detach) -> step env -> get real 4D reward in {0,1}.
      3) Compare predicted reward (in [0,1]) to env's reward => BCE loss.
      4) Backprop, update the model.

      NOTE: if the environment returns strictly {0,1}, BCE is fine. 
            If it was continuous [0,1], BCE can still be used, or MSE.
    """
    bce = nn.BCELoss()
    mse = nn.MSELoss()
    model.train()
    model.to(device)

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        total_samples = 0
        
        for batch in dataloader:
            # get input
            input_data = batch["input"].to(device)  # [B, 1, 24]
            input_data = input_data.squeeze(1)      # [B, 24]

            # forward pass
            pred_action, pred_reward = model(input_data)  # [B,16], [B,4] in [0,1]

            # detach actions to run environment
            actions_np = pred_action.detach().cpu().numpy()  # (B,16)

            # get environment rewards for each action
            env_rewards = []
            for i in range(actions_np.shape[0]):
                env.reset()
                # Possibly do more than 1 step, but here's 1 step:
                env.step(actions_np[i])  
                r4 = env.compute_4d_reward()  # [4] in {0,1}
                env_rewards.append(r4)

            env_rewards = torch.tensor(env_rewards, dtype=torch.float32, device=device)  # [B,4]

            # compare predicted reward vs. environment reward
            optimizer.zero_grad()
            loss_reward = bce(pred_reward, env_rewards)
            loss_reward.backward()
            # Optionally: clip gradients or scale them
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = input_data.size(0)
            total_loss += loss_reward.item() * bs
            total_samples += bs

        avg_loss = total_loss / total_samples
        if epoch % 5 == 0:
            print(f"[Phase 2] Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")


# process the gt with this clip result:
def LEAPsim_limits(type="regular"):
    sim_min = np.array([
        -1.047, -0.314, -0.506, -0.366,
        -1.047, -0.314, -0.506, -0.366,
        -1.047, -0.314, -0.506, -0.366,
        -0.349, -0.470, -1.200, -1.340
    ])
    sim_max = np.array([
        1.047, 2.230, 1.885, 2.042,
        1.047, 2.230, 1.885, 2.042,
        1.047, 2.230, 1.885, 2.042,
        2.094, 2.443, 1.900, 1.880
    ])
    return sim_min, sim_max



###########################################################
# 5) MAIN DEMO
###########################################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the MuJoCo environment
    xml_file = "/home/haozhe/Dropbox/physics/_data/leap_hand/scene_U.xml"
    gt_mesh_path='/home/haozhe/U.ply'
    gt_control_path='/home/haozhe/Dropbox/imitationlearning/U_Pick_and_Place/hand_data_hamer_distilled/leap_batched_output.npy'
    env = HandEnv(model_xml_path=xml_file, action_dim=16, frame_skip=5, episode_length=20)

    # Create the model (20D output)
    model = GraspMLP(input_dim=24, hidden_dim=256, action_dim=16, reward_dim=2)

    # Phase 1 DataLoader
    phase1_dataset = Phase1Dataset(gt_mesh_path, gt_control_path)
    phase1_loader = DataLoader(phase1_dataset, batch_size=8, shuffle=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    ################################
    # PHASE 1: Supervised on GT (16 actions) + Reward=1
    ################################
    print("==== Phase 1: Training on GT Actions + Reward=1 ====")
    train_phase_1(model, phase1_loader, optimizer, epochs=30, device=device)

    ################################
    # PHASE 2: Predict action, run env, get real 4D reward
    ################################
    print("\n==== Phase 2: Train reward outputs via MuJoCo contact ====")
    # # Reuse the same dataset for input_data; we ignore GT actions now
    train_phase_2(
        model=model,
        dataloader=phase1_loader,
        env=env,
        optimizer=optimizer,
        epochs=20,
        device=device,

    )

    # train_phase_3(
    #     model=model,
    #     dataloader=phase1_loader,
    #     optimizer=optimizer,
    #     epochs=20,
    #     device=device,

    # )

    # Save the model
    model_save_path = "grasp_model_with_validation.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    print("Done!")


if __name__ == "__main__":
    main()




    


# so i need validation now


# and also the mass reward

# i already have the gt robotic hand pose 




# 3.25 is the 0 in this poses
#  


