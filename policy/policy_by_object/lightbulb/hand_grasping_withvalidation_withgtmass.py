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

        self.mj_data.efc_force[:] = 0.0
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

    def compute_object_contacts(self):
        """
        Returns the number of contacts currently involving the object
        (i.e. geoms whose name includes 'U').
        """
        contact_count = 1
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1, g2 = c.geom1, c.geom2
            n1 = mj.mj_id2name(self.mj_model, mj.mjtObj.mjOBJ_GEOM, g1)
            n2 = mj.mj_id2name(self.mj_model, mj.mjtObj.mjOBJ_GEOM, g2)
            if ("lball" in n1) or ("lball" in n2):
                contact_count += 1
        return contact_count


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
    def __init__(self, input_dim=24, hidden_dim=256, action_dim=16, reward_dim=2,force_dim=1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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

        self.force_head = nn.Sequential(
            nn.Linear(hidden_dim, force_dim),
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
        pred_force  = self.force_head(feat).mean(dim=1)
        return pred_actions, pred_reward,pred_force
    



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
    def __init__(self, gt_mesh_path, gt_control_path, num_samples=1000, input_dim=192):
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


        # Reward is fixed to [1,1,1,1]
        self.fixed_reward = torch.ones(2, device="cuda")
        self.gt_force_initial =torch.ones(1, device="cuda")
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
        sample_reward = self.fixed_reward  
        gt_force_initial = self.gt_force_initial       # [4]
        return {
            "input": sample_input,
            "gt_action": sample_action,
            "gt_reward": sample_reward,
            "gt_force_initial": gt_force_initial
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
            gt_force_initial = batch["gt_force_initial"].to(device)  # [B, 1]
            optimizer.zero_grad()
            pred_action, pred_reward,pred_force = model(input_data)  
            # shapes: [B,16], [B,4], each in [0,1] for reward

            # 1) action loss
            loss_action = mse(pred_action, gt_action)

            # 2) reward loss (we want it to match [1,1,1,1])
            loss_reward = bce(pred_reward, gt_reward)

            loss_force = mse(pred_force, gt_force_initial)

            loss = loss_action + loss_reward +loss_force
            loss.backward()
            optimizer.step()

            bs = input_data.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

        avg_loss = total_loss / total_samples
        if epoch % 10 == 0:
            print(f"[Phase 1] Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")


def train_phase_2(model, dataloader, env, optimizer,   object_mass=0.2, 
                  gravity=9.81, 
                  max_force=5.0,  epochs=100, device="cuda"):
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
            pred_action, pred_reward,pred_force = model(input_data)  # [B,16], [B,4] in [0,1]

            # detach actions to run environment
            actions_np = pred_action.detach().cpu().numpy()  # (B,16)

            # get environment rewards for each action
            env_rewards = []
            env_forces = []
            for i in range(actions_np.shape[0]):
                env.reset()
                # Possibly do more than 1 step, but here's 1 step:
                env.step(actions_np[i])  
                r4 = env.compute_4d_reward()  # [4] in {0,1}
                env_rewards.append(r4)
                num_contacts = env.compute_object_contacts()
                
                # ground-truth force
                gt_force = object_mass * gravity * num_contacts

                # optionally scale into [0,1] if using Sigmoid
                # e.g. if we assume the maximum force is ~5N
                # tune 'max_force' as needed
                force_scaled = gt_force / max_force
                # clamp at 1.0 if it exceeds
                force_scaled = min(force_scaled, 1.0)

                env_forces.append(force_scaled)
            
            env_forces = torch.tensor(env_forces, dtype=torch.float32, device=device).unsqueeze(-1)  # [B,1]

            # 4) MSE between pred_force and env_forces
            optimizer.zero_grad()
            loss_force = mse(pred_force, env_forces)*0.1

            env_rewards = torch.tensor(env_rewards, dtype=torch.float32, device=device)  # [B,4]

            # compare predicted reward vs. environment reward
            optimizer.zero_grad()
            loss_reward = bce(pred_reward, env_rewards)
            total_loss = loss_reward*0.8 + loss_force*0.3
            total_loss.backward()
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

def train_phase_3(model, 
                  dataloader, 
                  env, 
                  optimizer, 
                  object_mass=0.2, 
                  gravity=9.81, 
                  max_force=5.0, 
                  epochs=10, 
                  device="cuda"):
    """
    Phase 3:
      - We want model's predicted force to match 'mass*g*(num_obj_contacts)'.
      - We'll do a single environment step for each action (demo style).
      - We'll MSE on the difference between pred_force and actual_force (scaled).
    """
    mse = nn.MSELoss()
    model.train()
    model.to(device)

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            # 1) Input
            input_data = batch["input"].to(device)     # [B, 24]
            # We do NOT need gt_action/gt_reward for Phase 3,
            # just the input for the model.

            # 2) Forward
            pred_action, pred_reward, pred_force = model(input_data)  # shapes: [B,16], [B,2], [B,1]

            # 3) Detach predicted actions, step env => measure contact
            actions_np = pred_action.detach().cpu().numpy()  # (B,16)

            env_forces = []
            for i in range(actions_np.shape[0]):
                env.reset()
                # Step once (or multiple times if you prefer)
                env.step(actions_np[i])
                
                # measure #contacts with object
                num_contacts = env.compute_object_contacts()
                
                # ground-truth force
                gt_force = object_mass * gravity * num_contacts

                # optionally scale into [0,1] if using Sigmoid
                # e.g. if we assume the maximum force is ~5N
                # tune 'max_force' as needed
                force_scaled = gt_force / max_force
                # clamp at 1.0 if it exceeds
                force_scaled = min(force_scaled, 1.0)

                env_forces.append(force_scaled)
            
            env_forces = torch.tensor(env_forces, dtype=torch.float32, device=device).unsqueeze(-1)  # [B,1]

            # 4) MSE between pred_force and env_forces
            optimizer.zero_grad()
            loss_force = mse(pred_force, env_forces)*0.1

            loss_force.backward()
            optimizer.step()

            bs = input_data.size(0)
            total_loss += loss_force.item() * bs
            total_samples += bs

        avg_loss = total_loss / total_samples
        if epoch % 5 == 0:
            print(f"[Phase 3] Epoch {epoch}/{epochs}, ForceLoss: {avg_loss:.4f}")

###########################################################
# 5) MAIN DEMO
###########################################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the MuJoCo environment
    xml_file = "/home/haozhe/Dropbox/physics/_data/leap_hand/scene_lightbulb.xml"
    gt_mesh_path='/home/haozhe/lightbulb.ply'
    gt_control_path='/home/haozhe/Dropbox/imitationlearning/lightbulb_pick_place/leap_batched_output.npy'
    env = HandEnv(model_xml_path=xml_file, action_dim=16, frame_skip=5, episode_length=20)

    # Create the model (20D output)
    model = GraspMLP(input_dim=24, hidden_dim=256, action_dim=16, reward_dim=2,force_dim=1)

    # Phase 1 DataLoader
    phase1_dataset = Phase1Dataset(gt_mesh_path, gt_control_path,num_samples=1000, input_dim=192)
    phase1_loader = DataLoader(phase1_dataset, batch_size=8, shuffle=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    ################################
    # PHASE 1: Supervised on GT (16 actions) + Reward=1
    ################################
    print("==== Phase 1: Training on GT Actions + Reward=1 ====")
    train_phase_1(model, phase1_loader, optimizer, epochs=50, device=device)

    ################################
    # PHASE 2: Predict action, run env, get real 4D reward
    ################################
    print("\n==== Phase 2: Train reward outputs via MuJoCo contact ====")
    # Reuse the same dataset for input_data; we ignore GT actions now
    train_phase_2(
        model=model,
        dataloader=phase1_loader,
        env=env,
        optimizer=optimizer,
        epochs=10,
        device=device,
        object_mass=0.038,

    )
    # print("\n==== Phase 3: Train force output via object contact ====")
    # train_phase_3(model, phase1_loader, env, optimizer,
    #               object_mass=0.2, 
    #               gravity=9.81, 
    #               max_force=2.0,   # adjust if needed
    #               epochs=5, 
    #               device=device)

    # Save the model
    model_save_path = "lightbulb/grasp_model_with_validation_withmass_38g.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    print("Done!")

if __name__ == "__main__":
    main()




    
