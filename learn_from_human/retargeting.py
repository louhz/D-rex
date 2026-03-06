import os
import glob
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R

############################################################################
# 1) Your existing ManoRetargeter (unchanged)
############################################################################
from typing import Optional, Union
from manotorch.anchorlayer import AnchorLayer
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer, MANOOutput

class ManoRetargeter:
    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None):
        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            use_pca=False,
            side="left",
            center_idx=None,
            mano_assets_root="./manotorch/assets/mano",  # change as needed
            flat_hand_mean=False,
        )
        self.axis_layer = AxisLayerFK(mano_assets_root="./manotorch/assets/mano")  # change as needed

    def mano_retarget(self, joint_pose, shape_params):
        """
        joint_pose:  (48,) np.array or Torch => For your code, pass as np.
        shape_params: (10,) np.array or Torch => pass as np.
        """
        mano_results: MANOOutput = self.mano_layer(joint_pose.reshape(1,joint_pose.shape[0]), shape_params.reshape(1,shape_params.shape[0]))
        T_g_p = mano_results.transforms_abs  # (B=1, 16, 4, 4) for single sample
        T_g_a, R, ee = self.axis_layer(T_g_p)  # ee is (B=1,16,3)
        ee = ee.flatten().tolist()  # shape(16*3=48)
        output = self._get_poses(ee)
        return output

    def _get_poses(self, finger_joints):
        """
        Takes the (16,3) euler angles from axis_layer, extracts certain joints,
        and returns final angles (like your snippet).
        """
        finger_joints = np.reshape(finger_joints, (16, 3))
        finger_mcp_id = [1, 4, 10]
        finger_pip_id = [2, 5, 11]
        finger_dip_id = [3, 6, 12]

        ee_mcps = finger_joints[finger_mcp_id]
        ee_pips = finger_joints[finger_pip_id]
        ee_dips = finger_joints[finger_dip_id]

        joint_mcp_side    = -ee_mcps[:, 1]
        joint_mcp_forward =  ee_mcps[:, 2]
        joint_pip         =  ee_pips[:, 2]
        joint_dip         =  ee_dips[:, 2]

        thumb_cmc_side    = finger_joints[13:14, 1]
        thumb_cmc_forward = finger_joints[13:14, 2]
        thumb_mcp         = finger_joints[14:15, 2]
        thumb_ip          = finger_joints[15:, 2]

        output = []
        for i in range(3):
            output += [joint_mcp_side[i], joint_mcp_forward[i], joint_pip[i], joint_dip[i]]

        output += [
            thumb_cmc_side[0],
            thumb_cmc_forward[0],
            thumb_mcp[0],
            thumb_ip[0],
        ]
        return output

############################################################################
# 2) Dataset that treats each YAML file as one sample
############################################################################
class MultiFileManoDataset(Dataset):
    """
    Each .yaml file is one sample.
    Example of each file structure:
    {
      "betas":         <10 floats total or (1,10)>,
      "global_orient": <3x3 rotation matrix or (1,3,3)>,
      "hand_pose":     <16,3,3 rotation matrices or (1,16,3,3)>,
      ...
    }
    """
    def __init__(self, yaml_file_list):
        """
        Args:
            yaml_file_list (List[str]): a list of file paths to .yaml files.
        """
        super().__init__()
        self.yaml_file_list = yaml_file_list

    def __len__(self):
        return len(self.yaml_file_list)

    def __getitem__(self, idx):
        yaml_path = self.yaml_file_list[idx]
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # We'll assume each file has EXACTLY 1 sample inside (the usual case).
        # Convert to float32 arrays
        betas = np.array(data["betas"], dtype=np.float32)               # shape (10,) or (1,10)
        global_orient = np.array(data["global_orient"], dtype=np.float32)  # shape (3,3) or (1,3,3)
        hand_pose = np.array(data["hand_pose"], dtype=np.float32)       # shape (16,3,3) or (1,16,3,3)

        # If your YAML is nested or has shape (1,10) etc., you might need to squeeze.
        if betas.ndim == 2 and betas.shape[0] == 1:
            betas = betas[0]   # => (10,)
        if global_orient.ndim == 3 and global_orient.shape[0] == 1:
            global_orient = global_orient[0]  # => (3,3)
        if hand_pose.ndim == 4 and hand_pose.shape[0] == 1:
            hand_pose = hand_pose[0]          # => (16,3,3)

        # Return as torch Tensors
        return {
            "betas": torch.from_numpy(betas),            # (10,)
            "global_orient": torch.from_numpy(global_orient),  # (3,3)
            "hand_pose": torch.from_numpy(hand_pose)     # (16,3,3)
        }

def collate_fn(batch):
    """
    We treat each file as 1 sample -> we stack them along dim=0:
      - betas => (B,10)
      - global_orient => (B,3,3)
      - hand_pose => (B,16,3,3)
    """
    betas_list = []
    global_orient_list = []
    hand_pose_list = []

    for item in batch:
        betas_list.append(item["betas"])
        global_orient_list.append(item["global_orient"])
        hand_pose_list.append(item["hand_pose"])

    betas = torch.stack(betas_list, dim=0)               # (B,10)
    global_orient = torch.stack(global_orient_list, dim=0)  # (B,3,3)
    hand_pose = torch.stack(hand_pose_list, dim=0)       # (B,16,3,3)

    return {
        "betas": betas,
        "global_orient": global_orient,
        "hand_pose": hand_pose
    }

############################################################################
# 3) Converting rotation matrices -> axis-angle
############################################################################

def rotation_matrices_to_axis_angles(global_orient_mat, hand_pose_mat):
    """
    global_orient_mat: (B,3,3)  torch.Tensor
    hand_pose_mat:     (B,16,3,3)
    Returns:           (B,48)   # 3 + 15*3=48
    """

    global_orient_np = global_orient_mat.numpy()  # (B,3,3)
    hand_pose_np     = hand_pose_mat.numpy()      # (B,16,3,3)



    g_aa = R.from_matrix(global_orient_np).as_rotvec()   # (3,)
    h_aa = R.from_matrix(hand_pose_np).as_rotvec()       # (16,3)

        # Typically we use the first 15 finger joints => 45
        # plus 3 from global => total 48
    g_aa_3  = torch.from_numpy(g_aa).float()            # (3,)
    h_aa_15 = torch.from_numpy(h_aa[:15]).float()       # shape (15,3)

    pose_48_i = torch.cat([g_aa_3.reshape(-1), h_aa_15.reshape(-1)], dim=0)  # (48,)



    return pose_48_i

############################################################################
# 4) Example usage
############################################################################

if __name__ == "__main__":
    # 1) Gather all .yaml paths from a folder
    # For example, if you have multiple files: frame_0000_0mano_output.yaml, frame_0001_0mano_output.yaml, etc.
    folder_path = "./data/humandemonstration/screwdrivermanipulate/seq/1/pose"
    yaml_file_list = sorted(glob.glob(os.path.join(folder_path, "*.yaml")))
    # Filter to keep only the ones that end with "_1mano_output.yaml"
    filtered_files = [f for f in yaml_file_list if "_0mano_output.yaml" in os.path.basename(f)]

    # 2) Create the dataset
    dataset = MultiFileManoDataset(filtered_files)

    # 3) Create a DataLoader that loads them in one big batch
    #    If you want each file => 1 sample => entire dataset => 1 batch => set batch_size = len(dataset)
    #    If you prefer smaller batches, set e.g. batch_size=4 or batch_size=1, etc.

    # 4) We'll just do one iteration => which loads all items at once (since batch_size = len(dataset))
    mano_retargeter = ManoRetargeter()


    batched_output=[]
    leap_batched_output=[]
    for batch_idx in range(len(dataset)):
        batch=dataset[batch_idx]
        print(f"Batch {batch_idx}:")
        print("  betas shape:        ", batch["betas"].shape)         # (B,10)
        print("  global_orient shape:", batch["global_orient"].shape) # (B,3,3)
        print("  hand_pose shape:    ", batch["hand_pose"].shape)     # (B,16,3,3)

        # Convert => axis-angle => shape (B,48)
        pose_aa_48 = rotation_matrices_to_axis_angles(
            batch["global_orient"],
            batch["hand_pose"]
        )

        # shape_params => (B,10)
        shape_params = batch["betas"]  # shape (B,10)

        # Combine => (B,58)
        mano_grasp = torch.cat([pose_aa_48, shape_params])  # (B,48+10)= (B,58)
        batched_output.append(mano_grasp)
        # Because your retargeter is single-sample, let's loop over each item

            # shape (48,) and (10,)
        joint_pose_i  = mano_grasp[:48]
        shape_param_i = mano_grasp[48:]

            # retarget
        leap_hand_output = mano_retargeter.mano_retarget(joint_pose_i, shape_param_i)
        print(f"    => Leap hand output: {leap_hand_output}")
        leap_batched_output.append(leap_hand_output)

    # save leap_batched_output
    np.save('./data/humandemonstration/screwdrivermanipulate/leap_batched_output_1.npy',leap_batched_output)







