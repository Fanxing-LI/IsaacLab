# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check through the heterogeneous task manager that the objects of every task rest stably at their start poses.

Runs the heterogeneous VLA environment without cameras. In batches of ``--num_envs``, the task manager assigns one task
per environment, the environments are reset, and the arm is held at its default pose for ``--seconds`` while the
environments step without automatic resets. A task fails when an object on the furniture moves more than 3 cm, tilts
more than 20 degrees, is not finite, or when the task succeeds at once. An object that starts in a container to be
taken out may settle in it on its side and move up to 5 cm. Writes the failing task names to ``--output`` (JSON).

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/validate_heterogeneous_tasks.py --num_envs 128 --output failed.json
"""

import argparse
import json
import math

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="IsaacContrib-VLA-Heterogeneous-Franka")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--output", required=True)
parser.add_argument("--mesh_cache_dir", default=None, help="Directory of Newton's mesh cache, shared across runs.")
add_launcher_args(parser)
args = parser.parse_args()

import gymnasium as gym  # noqa: E402
import newton  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.contrib.heterogeneous_tasks import mdp  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

if args.mesh_cache_dir:
    newton.config.cache_dir_meshes = args.mesh_cache_dir
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
for cfg in (env_cfg, *(env_cfg.tasks or {}).values()):
    for name in ("over_shoulder_left_camera", "wrist_cam"):
        setattr(cfg.scene, name, None)
        setattr(cfg.observations.policy, name, None)

with launch_simulation(env_cfg, args):
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    manager = env.task_manager
    names = manager.task_names
    # Slot 0 holds the furniture; the others hold the objects on it.
    slots = [name for name in env.scene.rigid_objects if name.startswith("object_") and name != "object_0"]
    # The object each take-out task starts in a container, by slot.
    taken_out = {
        task: slots.index(term.params["objects"][0].name)
        for task, term in manager.task_terms("success").items()
        if term.func is mdp.object_outside_of
    }
    robot = env.scene["robot"]
    hold = torch.cat((robot.data.default_joint_pos.torch[:, :7], torch.ones(env.num_envs, 1, device=env.device)), 1)
    steps = math.ceil(args.seconds / env.step_dt)
    failed, report = [], {}
    for start in range(0, len(names), env.num_envs):
        tasks = torch.arange(start, min(start + env.num_envs, len(names)), device=env.device)
        envs = torch.arange(len(tasks), device=env.device)
        manager.set_task(envs, tasks)
        env.reset(env_ids=envs)
        before = torch.stack([env.scene[s].data.root_link_pose_w.torch[envs] for s in slots], 1)  # (E, S, 7)
        succeeded = torch.zeros(len(tasks), dtype=torch.bool, device=env.device)
        for _ in range(steps):
            env.step(hold, auto_reset=False)
            succeeded |= env.termination_manager.get_term("success")[envs]
        after = torch.stack([env.scene[s].data.root_link_pose_w.torch[envs] for s in slots], 1)
        moved = (after[..., :3] - before[..., :3]).norm(dim=-1)
        # The z axis of a quaternion (x, y, z, w) in the world.
        x, y, z, w = after[..., 3], after[..., 4], after[..., 5], after[..., 6]
        up = 1.0 - 2.0 * (x * x + y * y)
        tilted = up < math.cos(math.radians(20.0))
        # The hidden stand-ins of empty slots are parked far away.
        parked = (before[..., 1] - env.scene.env_origins[envs, None, 1]) < -40.0
        # Objects that start in a container to be taken out settle in it, possibly on their side.
        inside = torch.zeros_like(parked)
        for row, task in enumerate(tasks.tolist()):
            if task in taken_out:
                inside[row, taken_out[task]] = True
        unstable = (moved > torch.where(inside, 0.05, 0.03)) | (tilted & ~inside)
        bad = (unstable | ~torch.isfinite(after).all(dim=-1)) & ~parked
        for row, task in enumerate(tasks.tolist()):
            if bad[row].any() or succeeded[row]:
                failed.append(names[task])
                report[names[task]] = {
                    "moved_cm": [round(v * 100, 1) for v in moved[row].tolist()],
                    "up": [round(v, 2) for v in up[row].tolist()],
                    "succeeded": bool(succeeded[row]),
                }
        print(f"tasks {start}-{start + len(tasks) - 1}: {int((bad.any(1) | succeeded).sum())} failed", flush=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump({"failed": failed, "details": report}, file, indent=1)
    print("FAILED", len(failed), failed, flush=True)
    env.close()
