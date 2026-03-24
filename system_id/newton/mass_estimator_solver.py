"""
Newton rewrite of D-rex's `system_id/diff_obj/mass_estimator_cookie.py`.

What this ports from gradsim -> Newton
--------------------------------------
- `gradsim.forces.ConstantForce` -> direct world-frame rigid-body wrench writes into
  `state.body_f`.
- `gradsim.simulator.Simulator` / `gradsim.engines.*` ->
  `newton.solvers.SolverSemiImplicit.step(...)`.
- `gradsim.contacts` ground-plane logic -> `newton.CollisionPipeline(...).collide(...)`
  against a Newton ground plane.
- D-rex's per-vertex mass parameterization -> differentiable reduction to body
  mass / COM / inertia, written into `model.body_*` and `model.shape_transform`
  each forward pass.

Important note
--------------
This is a *faithful Newton port of the differentiable objective*, not a claim that
Newton and gradsim will be numerically identical. D-rex's original contact model is a
custom impulse-style plane contact integrator, while Newton uses its own rigid contact
pipeline/solver. The training loop, force-window semantics, contact-patch expansion,
and trajectory loss are preserved as closely as possible.

Expected inputs
---------------
You can keep your existing D-rex data loading/preprocessing and pass the resulting
arrays into `NewtonMassEstimatorCookie`:

- `vertices`: (N, 3) mesh vertices in the object's mesh-local frame.
- `faces`: (F, 3) triangle indices.
- `positions_gt`: (T, 3) target trajectory positions.
- `orientations_gt`: (T, 4) target quaternions in **xyzw** order. If your GT is
  **wxyz**, convert it before constructing the estimator.
- `contact_point_local`: (3,) contact point already expressed in the object's local
  frame (equivalent to D-rex's flipped/contact-local point before KNN lookup).
- `impulse_force_world`: (3,) the force vector you want to apply to *each* point of
  the KNN-expanded contact patch during the active force window.

The default settings match the structure of the D-rex script:
- uniform-density optimization enabled by default
- KNN-expanded contact patch
- trajectory pose loss over `compare_every` frames
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import warp as wp
import warp.optim

import newton


MASS_EPS = 1.0e-8
INERTIA_REG_EPS = 1.0e-6


def _as_numpy_f32(x: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Convert torch / list / numpy input to contiguous float32 numpy."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    if shape is not None:
        arr = arr.reshape(shape)
    return np.ascontiguousarray(arr)


def _as_numpy_i32(x: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.int32)
    if shape is not None:
        arr = arr.reshape(shape)
    return np.ascontiguousarray(arr)


def point_mass_properties_np(vertices: np.ndarray, masses: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Point-mass mass / COM / inertia used to seed the Newton body."""
    masses = np.maximum(np.asarray(masses, dtype=np.float32), MASS_EPS)
    vertices = np.asarray(vertices, dtype=np.float32)

    total_mass = float(np.sum(masses))
    total_mass = max(total_mass, MASS_EPS)
    com = np.sum(vertices * masses[:, None], axis=0) / total_mass

    rel = vertices - com[None, :]
    x = rel[:, 0]
    y = rel[:, 1]
    z = rel[:, 2]
    inertia = np.zeros((3, 3), dtype=np.float32)
    inertia[0, 0] = np.sum(masses * (y * y + z * z))
    inertia[1, 1] = np.sum(masses * (x * x + z * z))
    inertia[2, 2] = np.sum(masses * (x * x + y * y))
    inertia[0, 1] = inertia[1, 0] = -np.sum(masses * x * y)
    inertia[0, 2] = inertia[2, 0] = -np.sum(masses * x * z)
    inertia[1, 2] = inertia[2, 1] = -np.sum(masses * y * z)
    return total_mass, com.astype(np.float32), inertia.astype(np.float32)


def build_contact_patch_indices(vertices: np.ndarray, contact_points_local: np.ndarray, k: int = 1200) -> np.ndarray:
    """
    D-rex equivalent of `find_closest_vertices(..., k=1200)`.

    Returns a unique, flattened vertex-index set for the KNN-expanded contact patch.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    contact_points_local = np.asarray(contact_points_local, dtype=np.float32).reshape(-1, 3)

    kd_tree = cKDTree(vertices)
    _distances, closest_indices = kd_tree.query(contact_points_local, k=k)
    closest_indices = np.asarray(closest_indices, dtype=np.int32).reshape(-1)
    return np.unique(closest_indices)


@wp.kernel
def relu_masses_uniform(
    base_masses: wp.array(dtype=float),
    delta: wp.array(dtype=float),
    masses_out: wp.array(dtype=float),
):
    tid = wp.tid()
    masses_out[tid] = wp.max(base_masses[tid] + delta[0], MASS_EPS)


@wp.kernel
def relu_masses_nonuniform(
    base_masses: wp.array(dtype=float),
    delta: wp.array(dtype=float),
    masses_out: wp.array(dtype=float),
):
    tid = wp.tid()
    masses_out[tid] = wp.max(base_masses[tid] + delta[tid], MASS_EPS)


@wp.kernel
def reduce_point_mass_properties(
    vertices_mesh_local: wp.array(dtype=wp.vec3),
    vertex_masses: wp.array(dtype=float),
    vertex_count: int,
    total_mass_out: wp.array(dtype=float),
    com_out: wp.array(dtype=wp.vec3),
    inertia_out: wp.array(dtype=wp.mat33),
):
    if wp.tid() != 0:
        return

    total_mass = float(0.0)
    com = wp.vec3(0.0, 0.0, 0.0)

    for i in range(vertex_count):
        m = vertex_masses[i]
        total_mass = total_mass + m
        com = com + vertices_mesh_local[i] * m

    total_mass = wp.max(total_mass, MASS_EPS)
    com = com / total_mass

    I = wp.mat33(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    for i in range(vertex_count):
        m = vertex_masses[i]
        r = vertices_mesh_local[i] - com
        x = r[0]
        y = r[1]
        z = r[2]
        I = I + m * wp.mat33(
            y * y + z * z,
            -x * y,
            -x * z,
            -y * x,
            x * x + z * z,
            -y * z,
            -z * x,
            -z * y,
            x * x + y * y,
        )

    total_mass_out[0] = total_mass
    com_out[0] = com
    inertia_out[0] = I


@wp.kernel
def write_body_and_shape_properties(
    body_mass: wp.array(dtype=float),
    body_inv_mass: wp.array(dtype=float),
    body_com: wp.array(dtype=wp.vec3),
    body_inertia: wp.array(dtype=wp.mat33),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    shape_transform: wp.array(dtype=wp.transform),
    body_id: int,
    shape_id: int,
    total_mass: wp.array(dtype=float),
    com: wp.array(dtype=wp.vec3),
    inertia: wp.array(dtype=wp.mat33),
):
    if wp.tid() != 0:
        return

    m = wp.max(total_mass[0], MASS_EPS)
    c = com[0]
    I = inertia[0]
    I_reg = I + wp.mat33(
        INERTIA_REG_EPS,
        0.0,
        0.0,
        0.0,
        INERTIA_REG_EPS,
        0.0,
        0.0,
        0.0,
        INERTIA_REG_EPS,
    )

    # Match D-rex semantics: use the body frame as the COM frame, and move the
    # mesh shape by -COM inside the body. This keeps state.body_q's translation
    # equal to the COM/world pose just like the recentered gradsim body.
    body_mass[body_id] = m
    body_inv_mass[body_id] = 1.0 / m
    body_com[body_id] = wp.vec3(0.0, 0.0, 0.0)
    body_inertia[body_id] = I_reg
    body_inv_inertia[body_id] = wp.inverse(I_reg)
    shape_transform[shape_id] = wp.transform(-c, wp.quat_identity())


@wp.kernel
def apply_drex_style_external_wrenches(
    body_q: wp.array(dtype=wp.transform),
    body_f: wp.array(dtype=wp.spatial_vector),
    vertices_mesh_local: wp.array(dtype=wp.vec3),
    contact_patch_ids: wp.array(dtype=wp.int32),
    com: wp.array(dtype=wp.vec3),
    body_id: int,
    gravity_total_force_world: wp.vec3,
    impulse_force_per_patch_point_world: wp.vec3,
    contact_patch_count: int,
    impulse_scale: float,
):
    if wp.tid() != 0:
        return

    q = body_q[body_id]
    rot = wp.transform_get_rotation(q)
    current = body_f[body_id]

    # Gravity in D-rex is applied as a uniform per-vertex force over all vertices.
    # When recentered about COM, that contributes zero torque and a known net force.
    net_force = gravity_total_force_world
    net_torque = wp.vec3(0.0, 0.0, 0.0)

    if impulse_scale > 0.0:
        # Preserve D-rex semantics: the *same* force vector is applied to every
        # vertex in the KNN-expanded contact patch, so the total linear force is
        # patch_count * force and the torque is the sum over all patch points.
        net_force = net_force + impulse_force_per_patch_point_world * float(contact_patch_count)
        c = com[0]
        for i in range(contact_patch_count):
            vid = contact_patch_ids[i]
            r_body = vertices_mesh_local[vid] - c
            r_world = wp.quat_rotate(rot, r_body)
            net_torque = net_torque + wp.cross(r_world, impulse_force_per_patch_point_world)

    body_f[body_id] = current + wp.spatial_vector(net_force, net_torque)


@wp.kernel
def accumulate_pose_loss(
    body_q: wp.array(dtype=wp.transform),
    gt_positions: wp.array(dtype=wp.vec3),
    gt_orientations_xyzw: wp.array(dtype=wp.quat),
    body_id: int,
    frame_idx: int,
    loss: wp.array(dtype=float),
    pos_weight: float,
    quat_weight: float,
):
    if wp.tid() != 0:
        return

    q = body_q[body_id]
    pos = wp.transform_get_translation(q)
    quat = wp.transform_get_rotation(q)

    gt_pos = gt_positions[frame_idx]
    gt_quat = gt_orientations_xyzw[frame_idx]

    dp = pos - gt_pos

    # Direct quaternion MSE to mirror the original D-rex objective. If your
    # Newton rollout flips quaternion sign relative to GT, replace this with a
    # sign-invariant loss min(||q-q*||^2, ||q+q*||^2).
    dq0 = quat[0] - gt_quat[0]
    dq1 = quat[1] - gt_quat[1]
    dq2 = quat[2] - gt_quat[2]
    dq3 = quat[3] - gt_quat[3]

    pos_term = wp.dot(dp, dp)
    quat_term = dq0 * dq0 + dq1 * dq1 + dq2 * dq2 + dq3 * dq3
    loss[0] = loss[0] + pos_weight * pos_term + quat_weight * quat_term




@wp.kernel
def scale_loss_in_place(loss: wp.array(dtype=float), scale: float):
    if wp.tid() != 0:
        return
    loss[0] = loss[0] * scale


@wp.kernel
def sgd_step_float(param: wp.array(dtype=float), grad: wp.array(dtype=float), lr: float):
    tid = wp.tid()
    param[tid] = param[tid] - lr * grad[tid]


@dataclass
class NewtonMassEstimatorConfig:
    dt: float = 0.002
    epochs: int = 150
    compare_every: int = 1
    contact_knn: int = 1200
    uniform_density: bool = True
    base_vertex_mass: float = 0.002
    lr: float = 5.0e-3
    gravity_total_force_world: tuple[float, float, float] = (0.0, -10.0, 0.0)
    impulse_direction_mask: tuple[float, float, float] = (1.0, 0.0, 1.0)
    pos_weight: float = 1.0
    quat_weight: float = 1.0
    ke: float = 5.0e3
    kd: float = 5.0e1
    kf: float = 0.0
    mu: float = 0.7
    soft_contact_margin: float = 10.0
    enable_contact_normal_gradients: bool = True
    device: str | None = None


class NewtonMassEstimatorCookie:
    """
    Newton port of the D-rex cookie mass estimator.

    Reuse your existing D-rex data-loading code, then instantiate this class
    with the prepared arrays and call `train()`.
    """

    def __init__(
        self,
        vertices: Any,
        faces: Any,
        positions_gt: Any,
        orientations_gt_xyzw: Any,
        raw_position: Any,
        contact_point_local: Any,
        impulse_force_world: Any,
        active_impulse_steps: int,
        config: NewtonMassEstimatorConfig | None = None,
        base_vertex_masses: Any | None = None,
    ) -> None:
        self.cfg = config or NewtonMassEstimatorConfig()
        self.device = wp.get_device(self.cfg.device) if self.cfg.device else None

        self.vertices_np = _as_numpy_f32(vertices, (-1, 3))
        self.faces_np = _as_numpy_i32(faces, (-1, 3))
        self.positions_gt_np = _as_numpy_f32(positions_gt, (-1, 3))
        self.orientations_gt_np = _as_numpy_f32(orientations_gt_xyzw, (-1, 4))
        self.raw_position_np = _as_numpy_f32(raw_position, (3,))
        self.contact_point_local_np = _as_numpy_f32(contact_point_local, (3,))
        self.impulse_force_world_np = _as_numpy_f32(impulse_force_world, (3,))
        self.impulse_force_world_np = self.impulse_force_world_np * _as_numpy_f32(
            self.cfg.impulse_direction_mask, (3,)
        )
        self.active_impulse_steps = int(active_impulse_steps)

        self.num_frames = int(self.positions_gt_np.shape[0])
        self.vertex_count = int(self.vertices_np.shape[0])

        if base_vertex_masses is None:
            base_vertex_masses = np.full(self.vertex_count, self.cfg.base_vertex_mass, dtype=np.float32)
        self.base_vertex_masses_np = _as_numpy_f32(base_vertex_masses, (self.vertex_count,))

        self.contact_patch_ids_np = build_contact_patch_indices(
            self.vertices_np,
            self.contact_point_local_np[None, :],
            k=self.cfg.contact_knn,
        ).astype(np.int32)
        self.contact_patch_count = int(self.contact_patch_ids_np.shape[0])

        self.vertices_wp = wp.array(self.vertices_np, dtype=wp.vec3, device=self.device)
        self.faces_wp = wp.array(self.faces_np.reshape(-1), dtype=wp.int32, device=self.device)
        self.positions_gt_wp = wp.array(self.positions_gt_np, dtype=wp.vec3, device=self.device)
        self.orientations_gt_wp = wp.array(self.orientations_gt_np, dtype=wp.quat, device=self.device)
        self.base_vertex_masses_wp = wp.array(self.base_vertex_masses_np, dtype=float, device=self.device)
        self.contact_patch_ids_wp = wp.array(self.contact_patch_ids_np, dtype=wp.int32, device=self.device)

        # Trainable parameterization mirrors the D-rex torch.nn.Module:
        # masses = relu(base_masses + update)
        if self.cfg.uniform_density:
            self.mass_update = wp.zeros(1, dtype=float, requires_grad=True, device=self.device)
        else:
            self.mass_update = wp.zeros(self.vertex_count, dtype=float, requires_grad=True, device=self.device)

        self.vertex_masses = wp.zeros(self.vertex_count, dtype=float, requires_grad=True, device=self.device)
        self.total_mass = wp.zeros(1, dtype=float, requires_grad=True, device=self.device)
        self.com = wp.zeros(1, dtype=wp.vec3, requires_grad=True, device=self.device)
        self.inertia = wp.zeros(1, dtype=wp.mat33, requires_grad=True, device=self.device)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True, device=self.device)

        self.gravity_total_force_world = wp.vec3(*self.cfg.gravity_total_force_world)
        self.impulse_force_world = wp.vec3(*self.impulse_force_world_np.tolist())

        # Seed a valid initial rigid body from the base mass distribution.
        m0, c0, I0 = point_mass_properties_np(self.vertices_np, self.base_vertex_masses_np)
        cookie_mesh = newton.Mesh(
            self.vertices_np,
            self.faces_np.reshape(-1),
            compute_inertia=False,
            is_solid=True,
        )

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
        body_xform = wp.transform(wp.vec3(*self.raw_position_np.tolist()), wp.quat_identity())
        self.body_id = builder.add_body(
            xform=body_xform,
            mass=float(m0),
            inertia=wp.mat33(*I0.reshape(-1).tolist()),
            com=wp.vec3(0.0, 0.0, 0.0),
            label="cookie",
            lock_inertia=True,
        )

        cookie_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=self.cfg.ke,
            kd=self.cfg.kd,
            kf=self.cfg.kf,
            mu=self.cfg.mu,
            is_solid=True,
        )
        self.shape_id = builder.add_shape_mesh(
            body=self.body_id,
            xform=wp.transform(wp.vec3(*(-c0).tolist()), wp.quat_identity()),
            mesh=cookie_mesh,
            cfg=cookie_cfg,
            label="cookie_mesh",
        )
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=self.cfg.ke,
                kd=self.cfg.kd,
                kf=self.cfg.kf,
                mu=self.cfg.mu,
            )
        )

        self.model = builder.finalize(requires_grad=True, device=self.device)
        self.model.set_gravity((0.0, 0.0, 0.0))

        self.solver = newton.solvers.SolverSemiImplicit(self.model, enable_tri_contact=False)
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            soft_contact_margin=self.cfg.soft_contact_margin,
            requires_grad=True,
            enable_contact_normal_gradients=self.cfg.enable_contact_normal_gradients,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.states = [self.model.state(requires_grad=True) for _ in range(self.num_frames + 1)]
        self.train_iter = 0
        self.loss_history: list[float] = []
        self.estimated_vertex_masses_history: list[np.ndarray] = []

    def _materialize_vertex_masses(self) -> None:
        if self.cfg.uniform_density:
            wp.launch(
                relu_masses_uniform,
                dim=self.vertex_count,
                inputs=[self.base_vertex_masses_wp, self.mass_update],
                outputs=[self.vertex_masses],
                device=self.device,
            )
        else:
            wp.launch(
                relu_masses_nonuniform,
                dim=self.vertex_count,
                inputs=[self.base_vertex_masses_wp, self.mass_update],
                outputs=[self.vertex_masses],
                device=self.device,
            )

    def _update_body_properties_from_masses(self) -> None:
        self._materialize_vertex_masses()
        wp.launch(
            reduce_point_mass_properties,
            dim=1,
            inputs=[
                self.vertices_wp,
                self.vertex_masses,
                self.vertex_count,
            ],
            outputs=[self.total_mass, self.com, self.inertia],
            device=self.device,
        )
        wp.launch(
            write_body_and_shape_properties,
            dim=1,
            inputs=[
                self.model.body_mass,
                self.model.body_inv_mass,
                self.model.body_com,
                self.model.body_inertia,
                self.model.body_inv_inertia,
                self.model.shape_transform,
                self.body_id,
                self.shape_id,
                self.total_mass,
                self.com,
                self.inertia,
            ],
            outputs=[],
            device=self.device,
        )

    def _reset_rollout(self) -> None:
        self.states[0] = self.model.state(requires_grad=True)

    def forward(self) -> wp.array:
        self.loss.zero_()
        self._update_body_properties_from_masses()
        self._reset_rollout()

        compare_frames: list[int] = []
        if self.cfg.compare_every > 0:
            compare_frames = list(range(self.cfg.compare_every - 1, self.num_frames, self.cfg.compare_every))

        for t in range(self.num_frames):
            state_in = self.states[t]
            state_out = self.states[t + 1]
            state_in.clear_forces()

            impulse_scale = 1.0 if t < self.active_impulse_steps else 0.0
            wp.launch(
                apply_drex_style_external_wrenches,
                dim=1,
                inputs=[
                    state_in.body_q,
                    state_in.body_f,
                    self.vertices_wp,
                    self.contact_patch_ids_wp,
                    self.com,
                    self.body_id,
                    self.gravity_total_force_world,
                    self.impulse_force_world,
                    self.contact_patch_count,
                    float(impulse_scale),
                ],
                outputs=[],
                device=self.device,
            )

            # Use Newton's differentiable rigid-contact path each step.
            self.collision_pipeline.collide(state_in, self.contacts)
            self.solver.step(state_in, state_out, self.control, self.contacts, self.cfg.dt)

            if t in compare_frames:
                wp.launch(
                    accumulate_pose_loss,
                    dim=1,
                    inputs=[
                        state_out.body_q,
                        self.positions_gt_wp,
                        self.orientations_gt_wp,
                        self.body_id,
                        t,
                        self.loss,
                        self.cfg.pos_weight,
                        self.cfg.quat_weight,
                    ],
                    outputs=[],
                    device=self.device,
                )

        if not compare_frames:
            wp.launch(
                accumulate_pose_loss,
                dim=1,
                inputs=[
                    self.states[-1].body_q,
                    self.positions_gt_wp,
                    self.orientations_gt_wp,
                    self.body_id,
                    self.num_frames - 1,
                    self.loss,
                    self.cfg.pos_weight,
                    self.cfg.quat_weight,
                ],
                outputs=[],
                device=self.device,
            )
        else:
            wp.launch(
                scale_loss_in_place,
                dim=1,
                inputs=[self.loss, 1.0 / float(len(compare_frames))],
                outputs=[],
                device=self.device,
            )

        return self.loss

    def step(self) -> float:
        tape = wp.Tape()
        with tape:
            loss = self.forward()
        tape.backward(loss)

        if self.mass_update.grad is None:
            raise RuntimeError("mass_update.grad is None after backward().")

        wp.launch(
            sgd_step_float,
            dim=len(self.mass_update),
            inputs=[self.mass_update, self.mass_update.grad, self.cfg.lr],
            outputs=[],
            device=self.device,
        )

        loss_value = float(self.loss.numpy()[0])
        self.loss_history.append(loss_value)
        self.estimated_vertex_masses_history.append(self.current_vertex_masses())

        tape.zero()
        self.train_iter += 1
        return loss_value

    def train(self, epochs: int | None = None, verbose: bool = True) -> dict[str, Any]:
        epochs = self.cfg.epochs if epochs is None else int(epochs)
        for epoch in range(epochs):
            loss_value = self.step()
            if verbose:
                print(f"[Newton][epoch={epoch:04d}] loss={loss_value:.6f}")

        # Re-roll forward once with the final updated parameters so the returned
        # trajectory matches the final mass estimate, not the pre-update last step.
        self.forward()
        traj_pos, traj_quat = self.extract_trajectory()
        return {
            "loss_history": np.asarray(self.loss_history, dtype=np.float32),
            "estimated_vertex_masses": self.current_vertex_masses(),
            "trajectory_positions": traj_pos,
            "trajectory_orientations_xyzw": traj_quat,
            "contact_patch_ids": self.contact_patch_ids_np.copy(),
        }

    def current_vertex_masses(self) -> np.ndarray:
        self._materialize_vertex_masses()
        return np.asarray(self.vertex_masses.numpy(), dtype=np.float32).copy()

    def extract_trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        """Return rollout trajectory after the most recent forward/train step."""
        body_q = np.stack([state.body_q.numpy()[self.body_id] for state in self.states[1:]], axis=0)
        positions = body_q[:, 0:3].astype(np.float32)
        quats_xyzw = body_q[:, 3:7].astype(np.float32)
        return positions, quats_xyzw

    def inspect_differentiable_contacts(self) -> dict[str, np.ndarray] | None:
        """
        Access the new differentiable rigid-contact outputs from Newton PR #2164 / the
        merged equivalent API.
        """
        if getattr(self.contacts, "rigid_contact_diff_distance", None) is None:
            return None
        return {
            "distance": np.asarray(self.contacts.rigid_contact_diff_distance.numpy()).copy(),
            "normal": np.asarray(self.contacts.rigid_contact_diff_normal.numpy()).copy(),
            "point0_world": np.asarray(self.contacts.rigid_contact_diff_point0_world.numpy()).copy(),
            "point1_world": np.asarray(self.contacts.rigid_contact_diff_point1_world.numpy()).copy(),
        }


# -----------------------------------------------------------------------------
# Example wiring from the original D-rex script
# -----------------------------------------------------------------------------
#
# The idea is to keep your existing preprocessing from `mass_estimator_cookie.py`
# that produces:
#   - vertices / faces
#   - positions_gt / orientations_gt
#   - raw_position
#   - contact_points transformed into the object's local frame
#   - smooth force vector and active impulse duration
#
# Then replace the gradsim block with something like:
#
#   cfg = NewtonMassEstimatorConfig(
#       dt=dt,
#       epochs=args.epochs,
#       compare_every=args.compare_every,
#       uniform_density=True,
#       contact_knn=1200,
#       lr=5e-3,
#   )
#
#   estimator = NewtonMassEstimatorCookie(
#       vertices=vertices[0],
#       faces=faces[0],
#       positions_gt=positions_gt,
#       orientations_gt_xyzw=orientations_gt,
#       raw_position=raw_position,
#       contact_point_local=contact_points_local,
#       impulse_force_world=force_obj,
#       active_impulse_steps=active_impulse_step,
#       config=cfg,
#       base_vertex_masses=np.full(vertices.shape[1], 0.002, dtype=np.float32),
#   )
#
#   results = estimator.train()
#
# If your GT quaternions are stored as wxyz, convert them to xyzw before passing
# them into the estimator.
