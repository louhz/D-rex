import torch
from torch.utils.data import DataLoader

# Make sure to import the same classes/definitions you used during training
# from your own code/modules:
#  - GraspMLP
#  - HandEnv
#  - Phase1Dataset

from hand_grasping_withvalidation_withgtmass import GraspMLP, HandEnv, Phase1Dataset

import numpy as np

# clip prediecd actions to the range 



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



def run_inference():
    # ------------------------------------------------
    # 1) Setup device
    # ------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference on device: {device}")

    # ------------------------------------------------
    # 2) Recreate model and load checkpoint
    # ------------------------------------------------
    # Must match the same network definition used at training time
    model = GraspMLP(input_dim=24, hidden_dim=256, action_dim=16, reward_dim=2,force_dim=1)
    model_save_path = "grasp_model_with_validation_withmass_126g.pth"
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded model weights from {model_save_path}")

    # ------------------------------------------------
    # 3) (Optional) Recreate environment
    # ------------------------------------------------
    # If you want to actually test the predicted actions in MuJoCo:
    xml_file = "/home/haozhe/Dropbox/physics/_data/leap_hand/scene_U.xml"
    env = HandEnv(
        model_xml_path=xml_file, 
        action_dim=16,
        frame_skip=5, 
        episode_length=20
    )

    # ------------------------------------------------
    # 4) Prepare inference data
    # ------------------------------------------------
    # For example, if you want to run on new data, you can
    # create a dataset or just a single input.  
    # Here, we’ll reuse the Phase1Dataset or pick one sample from it:
    gt_mesh_path = '/home/haozhe/Dropbox/physics/_data/leap_hand/assets/U.ply'
    gt_control_path = '/home/haozhe/Dropbox/imitationlearning/U_Pick_and_Place/hand_data_hamer_distilled/leap_batched_output.npy'
    inference_dataset = Phase1Dataset(
        gt_mesh_path, 
        gt_control_path, 
        num_samples=10,       # smaller for inference
        input_dim=192
    )
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

    # ------------------------------------------------
    # 5) Run inference loop
    # ------------------------------------------------
    print("Running inference on new data...")
    for idx, input_data in enumerate(inference_loader):

        input_data_24 = input_data['input'] # adapt if needed

        input_data_24 = input_data_24.to(device)
        
        with torch.no_grad():
            # Output shape: [batch_size, 20]
            output = model(input_data_24)

        # We assume the first 16 dims are actions, the last 4 are the predicted reward or reward components
        pred_actions = output[0].view(16,)  # Reshape to [batch_size, 16]
        pred_reward = output[1].view(2,)   # 4D reward if that’s how your network is structured
        pred_force = output[2].view(1,)   # 4D reward if that’s how your network is structured
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
        in_range_data_1 = gtpose1 - 3.14
        in_range_data_2 = gtpose2 - 3.14
        in_range_data_3 = gtpose3 - 3.14

        print(f"Sample {idx}:")
        print("  Predicted actions:", pred_actions.cpu().numpy())
        print("  Predicted reward:", pred_reward.cpu().numpy())
        print("  GT pose 1:", in_range_data_1)
        print("  GT pose 2:", in_range_data_2)
        print("  GT pose 3:", in_range_data_3)
        # ------------------------------------------------
        # 6) (Optional) Step the environment
        # ------------------------------------------------
        # If you want to test the predicted actions in the real environment:
        env.reset()
        # The env expects a 16D action (float), you might pass it as a numpy array
        action_numpy = pred_actions.cpu().numpy()
        # action_numpy = in_range_data_1

        # clip this action_numpy
        print('predicted force', pred_force.cpu().numpy())
        # action_numpy = np.clip(action_numpy, -0.6, 0.6)
        obs, reward = env.step(action_numpy)
        # action_numpy=np.zeros(16)
        env.run_with_renderer(action_numpy)
        
        # save action_numpy to a txt file
        np.savetxt("action_numpy.txt", action_numpy, delimiter=",")
        print("  Env step reward:", reward)
        print("  Env next obs shape:", obs.shape if hasattr(obs, 'shape') else type(obs))

        # Stop after one sample if you just want a quick test
        # break  

    print("Inference complete.")

if __name__ == "__main__":
    run_inference()