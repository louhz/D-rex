# D-REX

Official repository for the ICLR 2026 paper:

**D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping**  
Haozhe Lou · Mingtong Zhang · Haoran Geng · Hanyang Zhou · Sicheng He · Zhiyuan Gao · Siheng Zhao · Jiageng Mao · Pieter Abbeel · Jitendra Malik · Daniel Seita · Yue Wang

**Project page:** https://robot-drex-engine.github.io/  
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
