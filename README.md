# D-REX

Official repository for the ICLR 2026 paper:

**D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping**  
Haozhe Lou · Mingtong Zhang · Haoran Geng · Hanyang Zhou · Sicheng He · Zhiyuan Gao · Siheng Zhao · Jiageng Mao · Pieter Abbeel · Jitendra Malik · Daniel Seita · Yue Wang

**Project page:** https://louhz.github.io/drex.github.io/
**Paper (OpenReview):** https://openreview.net/forum?id=13jshGCK9i


---

![D-REX Teaser](https://robot-drex-engine.github.io/static/images/teaser.png)

## TL;DR

D-REX is a **differentiable real-to-sim-to-real engine** for dexterous manipulation that:
- reconstructs object geometry from real videos using **Gaussian Splat** representations,
- performs **end-to-end mass identification** through a **differentiable physics engine** using real observations and robot interaction data,
- enables **force-aware grasping policy learning** by transferring feasible **human video demonstrations** into simulated robot demonstrations, improving sim-to-real performance.

**Keywords:** Real-to-Sim-to-Real · Differentiable Simulation · Learning Robotic Policies from Videos · System Identification

---

## What’s in this repository

This repo is intended to support:
- **Real-to-Sim** reconstruction (Gaussian Splats + simulation-ready assets)
- **Mass identification** via differentiable physics (system identification from interaction data)
- **Human-to-robot demonstration transfer** (from human videos to feasible robot trajectories in sim)
- **Force-aware policy learning** for dexterous grasping and manipulation

If you are looking for the algorithmic overview and results videos, start with:
- Project page: https://robot-drex-engine.github.io/
- OpenReview paper: https://openreview.net/forum?id=13jshGCK9i

---

## Method overview

Our pipeline can be viewed as four connected components:

1. **Real-to-Sim**
   - Capture videos of the scene / object.
   - Reconstruct geometry with Gaussian Splat representations.
   - Build simulation assets (e.g., collision / simulation proxies) for a digital twin.

2. **Learning from Human Demonstrations**
   - Collect limited human demonstrations from videos.
   - Convert/transfer these to feasible robot demonstrations in simulation.

3. **Mass Identification (Differentiable Engine)**
   - Execute robot interactions (e.g., pushing) and use real observations + robot signals.
   - Optimize object mass through a differentiable simulation objective to obtain physically plausible digital twins.

4. **Policy Learning**
   - Train a grasping/manipulation policy conditioned on the identified mass.
   - Use force-aware constraints/objectives to reduce sim-to-real failures caused by mass mismatch.

![Method](https://robot-drex-engine.github.io/static/images/method.png)

---

## Results highlights

### Mass identification across objects
D-REX demonstrates robust mass identification across diverse object geometries and mass values, supporting physically plausible digital twins.

![Mass identification objects](https://robot-drex-engine.github.io/static/images/mass_objects.png)

### Force-aware grasping improves robustness
Mass estimates enable **mass-conditioned, force-aware policies** that achieve stronger real-world grasping performance, especially for heavier objects where baselines degrade.

![Grasping success rates](https://robot-drex-engine.github.io/static/images/success_rate_grasping.png)

The project page includes additional qualitative results:
- grasping examples across multiple objects (e.g., household items with different mass/shape),
- mass-policy sensitivity experiments showing success drops when train/eval masses do not match,
- more dexterous tasks beyond basic grasping (e.g., tool-like objects and everyday interactions).

---






## Getting started

### 1) Clone
```bash
git clone <THIS_REPO_URL>
cd D-rex
```

## Install submodules

### Manotorch

Please follow the https://github.com/lixiny/manotorch for the installtion and checkpoint download
And put the downloaded checkpoint to the proper path for learn from human retargeting
```bash
mano_assets_root="./manotorch/assets/mano"
```


### Mcc-ho

### colmap
Please follow the https://github.com/colmap/colmap for install

### Gsplat 


You can find the example for rendering and reconstruction in 
https://github.com/louhz/robogs
And the download instruction. Only minimum code for physically consistent rendering is added in this repo



## Dataset

### Human demonstration
https://www.dropbox.com/scl/fo/xexfduqcxrnhtrirl5r8g/AFgcoI22K2MwssvtutWUv_A?rlkey=vvtqbj5kzsmjmpq40v4lj6raq&st=t8yhcjfd&dl=0

### Real2sim Reconstruction
https://www.dropbox.com/scl/fo/cnla4b164gff7bdiswbdx/AKFDFTeI7I_oL9NLA5vQCNo?rlkey=j2hn2twi7uave36r64gi8gwyt&st=cu38u7p0&dl=0

### Mass Identification
https://www.dropbox.com/scl/fo/nccdsq367ydvj7cw57rgx/AIRh1sTwnVpcQmBu2Hz8ako?rlkey=cn0b1a229ykbt9a1yqdkztwdc&st=ziyghiem&dl=0






For the real2sim dataset you are input the 360 degree video and obtain the final asset

For the human demonstration you will have the grasping demonstration and have the retargeted robotic motion


For mass identification, you will input control signal , asset from real2sim and the object trajectory by foundation pose, obtain the final mass



## Policy learning and sim2real
Then use the learned parameter and retargeted control signal for learning and deploy to real world leap hand. 



you can put this data in the proper place 



## Implementation details and current method drawbacks:

1: For the human hand retargeting, the Hamor only learns the relative scale instead of the absolute scale, thus, we need manually align and fix the human hand scale gap for different human manipulation data source, we hope this can be solved by following word. 

2: For the system identification, the mj_model and mj_data cannot save the running process information, thus, we decide to use a txt to save the rollout information and load it again for system identification. A smarter engineering trick can help for the code clearance and optimization speed.

3: The crop and obtain of the object mesh and gaussian splat can be finished by SAM3D with accurete depth scale, we never test this method but it is promising to test this for future data collection and system work.

# Todo

replace path



rewrite config


clean up readme and comment




