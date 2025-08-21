from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional, Sequence

from assign import AssignConfig
from constraints import LMConfig


class LMOptimizer:
    def __init__(self, skel, assigner, geom):
        """
        skel     – Skeleton model (with fk, joint limits, etc.)
        assigner – Marker-to-bone assigner
        geom     – Geometry module for residual computation
        """
        self.skel = skel
        self.assigner = assigner
        self.geom = geom

    # ------------------------
    # Helpers
    # ------------------------
    def _make_init_dof_indices(self) -> List[int]:
        idxs: List[int] = []
        J = self.skel.JOINT_IDX
        for jn in ("pelvis", "spine_top", "neck_top"):
            base = 3 * J[jn]
            idxs.extend([base+0, base+1, base+2])
        for jn in ("right_shoulder", "left_shoulder", "right_hip", "left_hip"):
            base = 3 * J[jn]
            idxs.extend([base+0, base+2])  # yaw + roll
        return sorted(set(idxs))

    def _split_delta(
        self,
        delta_full: np.ndarray,
        off: int,
        n_bones: int,
        cfg: LMConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split LM delta into joint, bone, and root components."""
        dth = delta_full[:off]
        off2 = off + n_bones
        dbl = delta_full[off:off2]
        droot = delta_full[off2:off2 + (3 if cfg.optimize_root else 0)]
        return dth, dbl, droot

    def _clip_steps(
        self,
        dth: np.ndarray,
        dbl: np.ndarray,
        droot: np.ndarray,
        cfg: LMConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Clip LM step sizes."""
        dth = np.clip(dth, -cfg.angle_step_clip, cfg.angle_step_clip)
        dbl = np.clip(dbl, cfg.bone_clip[0], cfg.bone_clip[1])
        if droot is not None and len(droot) > 0:
            droot = np.clip(droot, -0.5, 0.5)  # arbitrary safeguard
        return dth, dbl, droot

    # ------------------------
    # Main solver
    # ------------------------
    def fit(
        self,
        bone_lengths: Dict[str, float],
        theta_init: np.ndarray,
        markers: np.ndarray,
        joint_limits: Tuple[np.ndarray, np.ndarray],
        root_init: Optional[np.ndarray] = None,
        cfg: Optional[LMConfig] = None,
        rng_seed: int = 0,
    ):
        """
        LM solver for skeleton fitting.
        Returns (theta, bone_lengths_final, root, theta_hist, bl_hist, root_hist)
        """
        cfg = cfg or LMConfig()
        rng = np.random.default_rng(rng_seed)

        theta = theta_init.copy()
        bl_vec = np.array([bone_lengths[k] for k in cfg.bone_opt_keys], dtype=float)
        root = root_init if root_init is not None else np.zeros(3)

        lower, upper = joint_limits
        active = cfg.active_dof_indices
        bl_opt_keys = cfg.bone_opt_keys

        lm_lambda = cfg.lm_lambda0
        angles_hist, bl_hist, root_hist = [], [], []

        # initial assignment
        jp, _ = self.skel.fk(bone_lengths, theta, root)
        assign = self.assigner.assign(markers, jp, prev_state=None,
                                      bone_lengths=bone_lengths,
                                      geom=cfg.geom,
                                      bone_radii=cfg.bone_radii,
                                      cfg=cfg.assign)

        # residual function
        def residual(jp_local, idx_batch):
            rs = self.geom.residuals_hard(jp_local, markers,
                                          assign['hard'],
                                          geom=cfg.geom,
                                          bone_radii=cfg.bone_radii)
            return rs.reshape(-1)

        e = residual(jp, None)

        for it in range(cfg.max_iters):
            # build Jacobian and normal equations
            J = self.skel.jacobian(theta, bl_vec, root, active, bl_opt_keys)
            A = J.T @ J + lm_lambda * np.eye(J.shape[1])
            g = J.T @ e
            try:
                delta_full = np.linalg.solve(A, -g)
            except np.linalg.LinAlgError:
                break

            off = len(active)
            n_bones = len(bl_opt_keys)
            dth, dbl, droot = self._split_delta(delta_full, off, n_bones, cfg)

            improved = False
            prev_cost = 0.5 * float(e.T @ e)

            for s in cfg.line_search_scales:
                dth_s, dbl_s, droot_s = self._clip_steps(dth * s, dbl * s, droot * s, cfg)

                th_new = theta.copy()
                th_new[active] += dth_s
                th_new = np.minimum(np.maximum(th_new, lower), upper)
                bl_new = bl_vec + dbl_s
                r_new = root + (droot_s if cfg.optimize_root else 0.0)

                jp_new, bl_all_new = self.skel.fk_local(th_new, bl_new, r_new)

                assign_eval = assign
                if cfg.allow_trial_reassign and (cfg.auto_assign_bones or (cfg.marker_bones is None)):
                    assign_eval = self.assigner.assign(
                        markers, jp_new, prev_state=assign.get("state"),
                        bone_lengths=bl_all_new, geom=cfg.geom,
                        bone_radii=cfg.bone_radii, cfg=cfg.assign
                    )

                e_new = residual(jp_new, None)

                if float(np.linalg.norm(e_new)) < float(np.linalg.norm(e)):
                    theta, bl_vec, root = th_new, bl_new, r_new
                    lm_lambda = max(lm_lambda / cfg.lm_lambda_factor, cfg.lm_lambda_min)
                    e = e_new
                    improved = True
                    break

            if not improved:
                lm_lambda = min(lm_lambda * cfg.lm_lambda_factor, cfg.lm_lambda_max)

            angles_hist.append(theta.copy())
            bl_hist.append(bl_vec.copy())
            root_hist.append(root.copy())

            if float(np.linalg.norm(e)) < cfg.tolerance:
                break

        bl_final = bone_lengths.copy()
        for k, v in zip(bl_opt_keys, bl_vec):
            bl_final[k] = float(v)

        return theta, bl_final, root, angles_hist, bl_hist, root_hist

    # ------------------------
    # Init scan
    # ------------------------
    def pick_best_start_frame_and_yaw(
        self,
        markers_seq: np.ndarray,
        bone_lengths: Dict[str, float],
        *,
        num_frames_to_scan: int = 30,
        yaw_seeds_deg: Sequence[int] = (0, 90, 180, 270),
        shoulder_roll_pairs_deg: Optional[Sequence[Tuple[float, float]]] = None,
        cfg: Optional[LMConfig] = None,
        iters: int = 10,
        rng_seed: int = 0,
        verbose: bool = False,
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Lightweight scan over early frames and yaw seeds; returns (best_fidx, theta, root)."""
        cfg = cfg or LMConfig(auto_assign_bones=True)
        if shoulder_roll_pairs_deg is None:
            shoulder_roll_pairs_deg = [(+80, -80)]
        T = min(len(markers_seq), int(num_frames_to_scan))
        if T == 0:
            raise RuntimeError("Empty sequence in pick_best_start_frame_and_yaw")

        lower, upper = self.skel.default_joint_limits()
        init_idxs = self._make_init_dof_indices()
        best = dict(cost=np.inf, fidx=0, theta=None, root=None)

        for fidx in range(T):
            m = markers_seq[fidx]
            root_guess = np.mean(m, axis=0)
            for yaw in yaw_seeds_deg:
                for (r_roll, l_roll) in shoulder_roll_pairs_deg:
                    th0 = self.skel.default_joint_angles()
                    J = self.skel.JOINT_IDX
                    th0[3*J['pelvis'] + 0] = np.deg2rad(yaw)
                    th0[3*J['right_shoulder'] + 2] = np.deg2rad(r_roll)
                    th0[3*J['left_shoulder'] + 2] = np.deg2rad(l_roll)

                    local_cfg = LMConfig(
                        active_dof_indices=init_idxs,
                        optimize_bones=False, optimize_root=True,
                        max_iters=iters, tolerance=1e-3,
                        geom="segment", auto_assign_bones=True,
                        assign=AssignConfig(topk=1, soft_sigma_factor=0.25, enable_gate=False)
                    )

                    th_fit, bl_fit, r_fit, *_ = self.fit(
                        bone_lengths, th0, m,
                        joint_limits=(lower, upper),
                        root_init=root_guess,
                        cfg=local_cfg,
                        rng_seed=1234 + 97*fidx
                    )

                    jp, _ = self.skel.fk(bone_lengths, th_fit, r_fit)
                    corr = self.assigner.assign(
                        m, jp, prev_state=None, bone_lengths=bone_lengths,
                        geom="segment", bone_radii=None,
                        cfg=AssignConfig(topk=1)
                    )
                    rs = self.geom.residuals_hard(jp, m, corr['hard'],
                                                  geom="segment",
                                                  bone_radii=None).reshape(-1, 3)
                    rmse = float(np.sqrt(np.mean(np.sum(rs*rs, axis=1))))

                    if verbose:
                        print(f"[init] f={fidx:03d} yaw={yaw:3d} roll=({r_roll:+.0f},{l_roll:+.0f}) -> {rmse:.4f}")
                    if rmse < best['cost']:
                        best.update(cost=rmse, fidx=fidx, theta=th_fit.copy(), root=r_fit.copy())

        if best['theta'] is None:
            best['theta'] = self.skel.default_joint_angles()
            best['root'] = np.mean(markers_seq[0], axis=0)

        return best['fidx'], best['theta'], best['root']
