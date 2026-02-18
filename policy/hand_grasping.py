import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import mujoco as mj
from torch.utils.data import Dataset, DataLoader

import mujoco.viewer as viewer
import open3d as o3d
import time
import sys


# this is the clip for each hand pose output
def LEAPsim_limits(type = "regular"):
    sim_min = np.array([-1.047, -0.314, -0.506, -0.366, -1.047, -0.314, -0.506, -0.366, -1.047, -0.314, -0.506, -0.366, -0.349, -0.47, -1.20, -1.34])
    sim_max = np.array([1.047,    2.23,  1.885,  2.042,  1.047,   2.23,  1.885,  2.042,  1.047,   2.23,  1.885,  2.042,  2.094,  2.443, 1.90,  1.88])
    return sim_min, sim_max


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
            # Adjust the name, position, or orientation as needed


        obj_joint_name = "U"
        pos = [0.095, -0.07, 0.1]
        quat = [0.0, 0.707, 0.707, 0.0]

        # Retrieve the joint ID for 'U'
        obj_joint_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_JOINT, obj_joint_name)
        obj_qpos_adr = self.mj_model.jnt_qposadr[obj_joint_id]
        obj_qvel_adr = self.mj_model.jnt_dofadr[obj_joint_id]

        # Set position (first 3) and orientation (next 4) in qpos
        self.mj_data.qpos[obj_qpos_adr : obj_qpos_adr + 3] = pos
        self.mj_data.qpos[obj_qpos_adr + 3 : obj_qpos_adr + 7] = quat

        # Zero out velocities for this joint
        # (For a free joint, there should be 6 DOFs in qvel, but confirm your model)
        dof_count = self.mj_model.jnt_dofadr[obj_joint_id + 1] - obj_qvel_adr \
            if (obj_joint_id + 1 < self.mj_model.njnt) else (self.mj_model.nv - obj_qvel_adr)
        for i in range(dof_count):
            self.mj_data.qvel[obj_qvel_adr + i] = 0.0

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
            if "plate" in n1 or "plate" in n2:
                table_contact = 1.0

            # r1: object “U”
            if "U" in n1 or "U" in n2:
                object_contact = 1.0

            # r3: special – both geoms outside the exclusion set
            if g1 not in _EXCLUDE_GEOMS and g2 not in _EXCLUDE_GEOMS:
                special_contact = 1.0

        # r2: many contacts (>=3)
        if contact_count >= 3:
            many_contacts = 1.0

        return np.array(
            [table_contact, object_contact, many_contacts, special_contact],
            dtype=np.float32,
        )

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
    Outputs 20D: first 16 for joint actions, last 4 for contact reward.
    Then applies a mean pooling across the batch dimension so that
    the final output is a single 20D vector.

    Example:
      Input shape: [batch_size, input_dim]
      Network output: [batch_size, 20]
      Pooled output: [20]
    """
    def __init__(self, input_dim=24, hidden_dim=256, output_dim=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): shape (batch_size, input_dim)

        Returns:
            pred_actions (torch.Tensor): shape (16,) after mean pooling
            pred_reward (torch.Tensor): shape (4,) after mean pooling
        """
        # Forward pass -> shape (batch_size, 20)
        out = self.net(x)
        
        # Mean pool across batch dimension -> shape (20,)
        out_mean = out.mean(dim=1)

        # Split into [16,] for actions, [4,] for reward
        pred_actions = out_mean[:,:16]
        pred_reward = out_mean[:,16:]
        
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
    def __init__(self,gt_mesh_path, gt_control_path, num_samples=1000, input_dim=192):
        super().__init__()
        self.gt_mesh_path = gt_mesh_path
        self.gt_control_path = gt_control_path
        mesh=o3d.io.read_triangle_mesh(self.gt_mesh_path)
        input_vertices = torch.from_numpy(np.asarray(mesh.vertices)).to(torch.float32).to('cuda')
        # Dummy data: random
        self.vertices = positional_encoding_3d_torch(input_vertices)
        self.gt_actions = (
            torch.from_numpy(np.load(self.gt_control_path))
            .to(torch.float32)          # or .float()
        )
        # Reward is fixed to [1,1,1,1]
        self.fixed_reward = torch.ones(4)
        self.num_samples = self.gt_actions.shape[0]
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Same input for every sample, but different GT action from the list
        sample_input  = self.vertices           # shape [192]
        sample_action = self.gt_actions[idx]      # shape [16]
        sample_reward = self.fixed_reward         # shape [4]
        return {
            "input": sample_input,
            "gt_action": sample_action,
            "gt_reward": sample_reward,
        }

###########################################################
# 4) TRAINING: PHASE 1 & PHASE 2
###########################################################
def train_phase_1(model, dataloader, optimizer, epochs=100, device="cuda"):
    """
    Phase 1:
      - pred_actions ~ gt_actions
      - pred_reward ~ [1,1,1,1]
    """
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()  # or MSELoss for the reward
    model.train()
    model.to(device)

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for batch in dataloader:
            input_data = batch["input"].to(device)           # [B, 192]
            gt_action = batch["gt_action"].to(device)        # [B, 16]
            gt_reward = batch["gt_reward"].to(device)        # [B, 4]

            optimizer.zero_grad()
            pred_action, pred_reward = model(input_data)     # [B,16], [B,4]
            
            # 1) action loss
            loss_action = mse(pred_action, gt_action)
            # 2) reward loss vs 1
            loss_reward = bce(pred_reward, gt_reward)
            loss = loss_action + loss_reward
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_data.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        if epoch % 10 == 0:
            print(f"[Phase 1] Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")


def train_phase_2(model, 
                  dataloader, 
                  env,
                  optimizer, 
                  epochs=100, 
                  device="cuda"):
    """
    Phase 2:
      1) Predict 16 actions & 4 reward from the model.
      2) Detach the predicted actions and send them to MuJoCo (env) 
         to get a real 4D reward from actual contact or conditions.
      3) Compare pred_reward to the environment reward with BCE (or MSE).
      4) Update the model's parameters (including the action portion, if desired).

    No freezing is done here. The predicted actions are detached 
    when passed to the environment, so there's no gradient flow 
    through the environment step. But the rest of the network 
    still updates, which can indirectly change the action outputs 
    in future forward passes.
    """
    import torch
    import torch.nn as nn
    bce = nn.BCEWithLogitsLoss()

    model.train()
    model.to(device)

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        
        for batch in dataloader:
            input_data = batch["input"].to(device)  # [B, 192]

            # 1) Forward pass: get predicted actions & reward
            pred_action, pred_reward = model(input_data)  # shapes [B,16], [B,4]

            # 2) Detach the actions for environment inference
            #    so we do not backprop through environment step
            actions_np = pred_action.detach().cpu().numpy()  # [B,16]

            # 3) Run environment with these actions to get real 4D reward
            env_rewards = []
            for i in range(actions_np.shape[0]):
                env.reset()
                env.step(actions_np[i])  # one or more steps
                r4 = env.compute_4d_reward()
                env_rewards.append(r4)

            env_rewards = torch.tensor(env_rewards, dtype=torch.float32, device=device)  # [B,4]

            # 4) Compare predicted reward to environment reward
            optimizer.zero_grad()
            loss_reward = bce(pred_reward, env_rewards)
            loss_reward.backward()
            optimizer.step()

            total_loss += loss_reward.item() * input_data.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        if epoch % 10 == 0:
            print(f"[Phase 2] Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")


###########################################################
# 5) MAIN DEMO
###########################################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the MuJoCo environment
    xml_file = "/home/haozhe/Dropbox/physics/_data/leap_hand/scene_U.xml"
    gt_mesh_path='/home/haozhe/Dropbox/physics/_data/leap_hand/assets/U.ply'
    gt_control_path='/home/haozhe/Dropbox/imitationlearning/U_Pick_and_Place/hand_data_hamer_distilled/leap_batched_output.npy'
    env = HandEnv(model_xml_path=xml_file, action_dim=16, frame_skip=5, episode_length=20)

    # Create the model (20D output)
    model = GraspMLP(input_dim=24, hidden_dim=256, output_dim=20)

    # Phase 1 DataLoader
    phase1_dataset = Phase1Dataset(gt_mesh_path, gt_control_path,num_samples=1000, input_dim=192)
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
    # Reuse the same dataset for input_data; we ignore GT actions now
    train_phase_2(
        model=model,
        dataloader=phase1_loader,
        env=env,
        optimizer=optimizer,
        epochs=30,
        device=device,

    )


    # Save the model
    model_save_path = "grasp_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    print("Done!")


if __name__ == "__main__":
    main()