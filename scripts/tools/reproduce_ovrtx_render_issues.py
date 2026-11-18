# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reproduce two OVRTX rendering issues on the heterogeneous VLA environment.

The environment runs a different task in every environment, on Newton physics, with two 448 x 252 RGB cameras that
OVRTX renders: the static ``over_shoulder_left_camera`` and the ``wrist_cam`` on the gripper. Every environment holds
all candidates of its switchable objects, of which its task shows a few. The arm rolls its last joint back and forth,
which moves the wrist camera.

Issue 1, slow scene initialization. OVRTX either clones the scene itself (``native``, selected by
``ISAAC_LAB_OVRTX_USE_OVSTAGE=0``) or renders an independent OVStage (``ovstage``, ``=1``, which this branch selects
for scenes with switchable assets). The script measures each scene in its own process: the time of creating the
environment up to its first images, most of which is the first native submission (a call of
``ovrtx.Renderer.step``), and the time of the later renders, where ``ovstage`` commits the stage on every submission.

Issue 2, repeated renders of lazy camera reads. A camera renders when its data is read. Isaac Lab submits every
render product on each native submission (``OVRTXRenderer.render_batch``, isaac-sim/IsaacLab#8116), because OVRTX
takes far longer for a render product that was left out of the submission before. Reading one camera therefore
renders all cameras, and reading every camera renders every camera once per camera. The pose of a camera that
follows a body (``wrist_cam``) reaches OVRTX only at that camera's own read, so a submission that is requested for
another camera renders it at the pose of its last read. Isaac Lab leaves these images unread. The script measures
three kinds of camera reads:

* ``batched``: the work-around of this branch, ``Camera._update_outdated_buffers``, which reads all due cameras in
  one submission.
* ``lazy``: Isaac Lab's lazy reads. The script counts the native submissions of an environment step, lists the
  cameras each was requested for and the cameras it rendered, reads the unread images back, and compares each with
  the image and the pose of the camera's own read.
* ``requested_only``: lazy reads whose submissions the script cuts to the requested render product, which shows the
  cost that makes Isaac Lab submit every render product.

The script first creates a small environment, which fills the caches of the kernels, the shaders and the assets, and
then measures ``native`` and ``ovstage``. Each time of a creation is one sample, and the samples of a scene differ by
minutes between runs. Writes to ``--output``: ``report.txt`` (the measurements of both scenes side by side, also
printed) and, per scene, ``result.json``, ``run.log``, the log of OVRTX and one image per camera, which shows, for
the first four environments, the unread image of another camera's submission, the image of the camera's own read,
and four times their absolute difference.

256 environments run every task once and take about 70 GiB of host memory and 36 GiB of GPU memory (128
environments: 45 GiB and 23 GiB); fewer environments run the first tasks. Keep a local copy of the task assets (see
:mod:`isaaclab_tasks.contrib.heterogeneous_tasks`): without one, every creation of the environment checks each asset
against its server, which takes minutes.

.. code-block:: bash

    uv run python scripts/tools/reproduce_ovrtx_render_issues.py --num_envs 256
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

SCENES = {"native": "0", "ovstage": "1"}
"""Value of ``ISAAC_LAB_OVRTX_USE_OVSTAGE`` that selects each scene."""
READS = {
    "batched": "batched reads (work-around of this branch: one submission for all due cameras)",
    "lazy": "lazy reads (Isaac Lab: one submission per camera, which renders all cameras)",
    "requested_only": "lazy reads, submissions cut to the requested camera",
}
"""Kinds of camera reads the script measures, with their titles in the report."""
WARMUP_STEPS = 5
"""Environment steps before every measurement."""
REQUESTED_ONLY_STEPS = 5
"""Timed environment steps of the ``requested_only`` reads, whose submissions take seconds."""
TRACE_STEPS = 4
"""Environment steps whose unread images are read back."""

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="IsaacContrib-VLA-Heterogeneous-Franka")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments, each with its own task.")
parser.add_argument("--scene", choices=tuple(SCENES), help="Scene to measure (default: both, one process each).")
parser.add_argument("--steps", type=int, default=50, help="Timed environment steps of the batched and lazy reads.")
parser.add_argument("--output", help="Directory of the results (default: logs/ovrtx_render_issues/<time>).")
add_launcher_args(parser)
args = parser.parse_args()
output = Path(args.output or f"logs/ovrtx_render_issues/{datetime.now():%Y-%m-%d_%H-%M-%S}")

if args.scene is not None:
    # The process that compares the scenes only starts their processes.
    import gymnasium as gym  # noqa: E402
    import torch  # noqa: E402
    import warp as wp  # noqa: E402
    from PIL import Image  # noqa: E402

    from isaaclab.sensors import Camera, CameraCfg, SensorBase  # noqa: E402

    import isaaclab_tasks  # noqa: E402, F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


class Probe:
    """Records the OVRTX calls of the camera reads: every native submission, every camera pose write and their times.

    ``wp.synchronize()`` runs before and after every timed call.
    """

    def __init__(self):
        self.seconds: dict[str, list[float]] = {}
        """Durations of the timed calls since the last :meth:`take`, by name [s]."""
        self.submissions: list[dict] = []
        """Native submissions since the last :meth:`take`: the render products the submission was requested for and
        those it rendered, the render time it advances, the cameras whose poses were written since the submission
        before it, and its duration [s]."""
        self.submit_requested_only = False
        """Whether a submission renders only the render products it was requested for, instead of all of them."""
        self.keep_unread = False
        """Whether to keep the written camera poses and to read back the images a submission was not requested for."""
        self.poses: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        """Last written positions [m] and orientations (x, y, z, w) of the cameras, by render product."""
        self.unread: dict[str, tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]] = {}
        """Last RGB images of the cameras from submissions that were not requested for them, each with the pose
        that was written last before its submission, by render product."""
        self._requested: list[str] = []
        self._written: list[str] = []
        self._buffers: dict[str, dict] = {}

    def install(self) -> None:
        """Wrap the calls of the OVRTX renderer, whose modules load once the launcher has registered OVRTX."""
        import ovrtx
        from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer
        from isaaclab_ov.stage import OvstageBackend

        def timed(owner: type, method: str, name: str) -> None:
            original = getattr(owner, method)

            def wrapper(*positional, **keywords):
                with self._timed(name):
                    return original(*positional, **keywords)

            setattr(owner, method, wrapper)

        timed(OVRTXRenderer, "prepare_stage", "prepare_stage")
        timed(OVRTXRenderer, "update_transforms", "transform_write")
        timed(OvstageBackend, "commit", "stage_commit")

        update_camera = OVRTXRenderer.update_camera

        def timed_update_camera(renderer, render_data, positions, orientations, intrinsics):
            with self._timed("pose_write"):
                update_camera(renderer, render_data, positions, orientations, intrinsics)
            self._written.append(render_data.render_product_path)
            if self.keep_unread:
                self.poses[render_data.render_product_path] = (positions.torch.clone(), orientations.torch.clone())

        OVRTXRenderer.update_camera = timed_update_camera

        render_batch = OVRTXRenderer.render_batch

        def timed_render_batch(renderer, render_data):
            self._requested = [data.render_product_path for data in render_data]
            with self._timed("render_batch"):
                render_batch(renderer, render_data)

        OVRTXRenderer.render_batch = timed_render_batch

        # Renderer.step is step_async(...).wait().fetch(), so every native submission passes through step_async.
        step_async = ovrtx.Renderer.step_async

        def counted_step_async(renderer, *positional, **keywords):
            call = dict(zip(("render_products", "delta_time"), positional), **keywords)
            if self.submit_requested_only:
                call["render_products"] = set(self._requested)
            self.submissions.append(
                {
                    "requested": self._requested,
                    "rendered": sorted(call["render_products"]),
                    "delta_time": call["delta_time"],
                    "poses_written": self._written,
                }
            )
            self._written = []
            return step_async(renderer, **call)

        ovrtx.Renderer.step_async = counted_step_async

        step = ovrtx.Renderer.step

        def timed_step(renderer, *positional, **keywords):
            with self._timed("native_step"):
                products = step(renderer, *positional, **keywords)
            self.submissions[-1]["seconds"] = self.seconds["native_step"][-1]
            return products

        ovrtx.Renderer.step = timed_step

        process_render_products = OVRTXRenderer._process_render_products

        def timed_process_render_products(renderer, render_data, products):
            with self._timed("read_back"):
                process_render_products(renderer, render_data, products)
            if not self.keep_unread:
                return
            # The renderer reads back only the requested products. Read the others into buffers of their own.
            requested = {data.render_product_path for data in render_data}
            for data in renderer._camera_render_data:
                path = data.render_product_path
                if path in requested:
                    continue
                buffers = self._buffers.setdefault(
                    path, {name: wp.zeros_like(buffer) for name, buffer in data.warp_buffers.items()}
                )
                info = dict(data.renderer_info)
                frame = max(products[path].frames, key=lambda frame: frame.end_time)
                renderer._process_render_frame(data, frame, buffers)
                data.renderer_info.update(info)
                wp.synchronize()
                self.unread[path] = (wp.to_torch(buffers["rgba"])[..., :3].clone(), self.poses.get(path))

        OVRTXRenderer._process_render_products = timed_process_render_products

    def take(self) -> dict:
        """Return the durations and submissions recorded since the last take, and forget them."""
        taken = {"seconds": self.seconds, "submissions": self.submissions}
        self.seconds, self.submissions = {}, []
        return taken

    @contextmanager
    def _timed(self, name: str):
        wp.synchronize()
        started = time.perf_counter()
        yield
        wp.synchronize()
        self.seconds.setdefault(name, []).append(time.perf_counter() - started)


def wrist_motion(hold: torch.Tensor):
    """Yield actions that hold the start pose and roll the last arm joint back and forth at a constant speed."""
    step = 0
    while True:
        action = hold.clone()
        # A triangle wave of 0.8 rad with a period of 24 steps.
        action[:, 6] += 1.6 / math.pi * math.asin(math.sin(2.0 * math.pi * step / 24.0))
        yield action
        step += 1


def time_steps(env, probe: Probe, actions, steps: int) -> dict:
    """Step the environments and return the medians of the camera reads of an environment step.

    Args:
        env: The heterogeneous VLA environment.
        probe: The record of the OVRTX calls.
        actions: The actions of the next steps.
        steps: Number of timed steps, which follow :data:`WARMUP_STEPS` steps.

    Returns:
        The time of an environment step and of its renders [ms], the native submissions per step, the cameras a
        submission is requested for and renders, and the times of the calls of a submission [ms].
    """
    for _ in range(WARMUP_STEPS):
        env.step(next(actions), auto_reset=False)
    taken = []
    for _ in range(steps):
        probe.take()
        wp.synchronize()
        started = time.perf_counter()
        env.step(next(actions), auto_reset=False)
        wp.synchronize()
        taken.append({"env_step": time.perf_counter() - started, **probe.take()})

    def per_step(name: str) -> list[float]:
        return [sum(step["seconds"].get(name, ())) for step in taken]

    def per_call(name: str) -> float | None:
        calls = [seconds for step in taken for seconds in step["seconds"].get(name, ())]
        return 1e3 * statistics.median(calls) if calls else None

    submissions = [call for step in taken for call in step["submissions"]]
    render = [
        sum(parts) for parts in zip(per_step("pose_write"), per_step("transform_write"), per_step("render_batch"))
    ]
    return {
        "env_step_ms": 1e3 * statistics.median(step["env_step"] for step in taken),
        "render_ms_per_env_step": 1e3 * statistics.median(render),
        "transform_write_ms_per_env_step": 1e3 * statistics.median(per_step("transform_write")),
        "submissions_per_env_step": len(submissions) / steps,
        "cameras_requested_per_submission": statistics.fmean(len(call["requested"]) for call in submissions),
        "cameras_rendered_per_submission": statistics.fmean(len(call["rendered"]) for call in submissions),
        "render_batch_ms": per_call("render_batch"),
        "native_step_ms": per_call("native_step"),
        "stage_commit_ms": per_call("stage_commit"),
        "read_back_ms": per_call("read_back"),
    }


def trace_lazy_reads(env, probe: Probe, cameras: dict[str, Camera], actions) -> list[dict]:
    """Record the submissions of a few environment steps and compare the unread images with the cameras' own.

    The image of a camera from a submission that was requested for another camera and the image of its own read show
    the same simulation state. They differ by the noise of the renderer, and by the camera pose when the pose was not
    written before the other camera's submission. Saves the two images of every camera to the output directory.

    Args:
        env: The heterogeneous VLA environment, whose observations read every camera.
        probe: The record of the OVRTX calls.
        cameras: The cameras by name.
        actions: The actions of the next steps.

    Returns:
        For each step, its submissions and, for each camera, the mean absolute difference of its two images [0-255]
        and the distance [m] and angle [deg] between the pose its unread image was rendered at and the pose of its
        own read.
    """
    names = {camera._render_data.render_product_path: name for name, camera in cameras.items()}
    probe.keep_unread = True
    # The poses written in this step are those the next step's unread images are rendered at.
    env.step(next(actions), auto_reset=False)
    steps = []
    for _ in range(TRACE_STEPS):
        probe.take()
        probe.unread.clear()
        env.step(next(actions), auto_reset=False)
        record = {"submissions": [], "cameras": {}}
        for call in probe.take()["submissions"]:
            record["submissions"].append(
                {
                    "requested": [names[path] for path in call["requested"]],
                    "rendered": [names[path] for path in call["rendered"]],
                    "poses_written_before": [names[path] for path in call["poses_written"]],
                }
            )
        for path, (unread, pose) in probe.unread.items():
            own = cameras[names[path]].data.output["rgb"].torch
            entry = {"image_difference": (unread.float() - own.float()).abs().mean().item()}
            if pose is not None:
                position, orientation = probe.poses[path]
                angle = 2.0 * torch.acos((orientation * pose[1]).sum(dim=1).abs().clamp(max=1.0))
                entry["pose_distance_m"] = (position - pose[0]).norm(dim=1).mean().item()
                entry["pose_angle_deg"] = torch.rad2deg(angle).mean().item()
            record["cameras"][names[path]] = entry
            # One row per environment: the unread image, the camera's own image and four times their difference.
            rows = slice(0, 4)
            difference = (unread[rows].int() - own[rows].int()).abs().mul(4).clamp(max=255).to(torch.uint8)
            grid = torch.cat((unread[rows], own[rows], difference), dim=2).flatten(0, 1)
            Image.fromarray(grid.cpu().numpy()).save(output / f"{names[path]}.png")
        steps.append(record)
    probe.keep_unread = False
    return steps


def report(results: dict[str, dict]) -> str:
    """Return the measurements of the scenes side by side."""

    def mean(result: dict, camera: str, key: str, scale: float = 1.0) -> float | None:
        values = [step["cameras"][camera][key] for step in result["trace"] if key in step["cameras"].get(camera, ())]
        return scale * statistics.fmean(values) if values else None

    rows = [
        ("Issue 1: scene initialization", None),
        ("environment creation up to the first images [s]", lambda r: r["initialization"]["environment_creation_s"]),
        ("  OVRTX stage preparation [s]", lambda r: r["initialization"]["prepare_stage_s"]),
        ("  native submissions of the first images [s]", lambda r: r["initialization"]["submissions_s"]),
        ("assignment of the tasks and reset [s]", lambda r: r["task_assignment_s"]),
        ("peak host memory [GiB]", lambda r: r["peak_rss_mib"] / 1024.0),
        ("render of all cameras in one submission [ms]", lambda r: r["batched"]["render_batch_ms"]),
        ("  native Renderer.step [ms]", lambda r: r["batched"]["native_step_ms"]),
        ("  stage commit [ms]", lambda r: r["batched"]["stage_commit_ms"]),
        ("  read-back of the images [ms]", lambda r: r["batched"]["read_back_ms"]),
        ("write of the object transforms per step [ms]", lambda r: r["batched"]["transform_write_ms_per_env_step"]),
        ("Issue 2: repeated renders of lazy camera reads", None),
    ]
    for reads, title in READS.items():
        rows += [
            (title, None),
            ("  native submissions per environment step", lambda r, k=reads: r[k]["submissions_per_env_step"]),
            ("  cameras a submission is requested for", lambda r, k=reads: r[k]["cameras_requested_per_submission"]),
            ("  cameras a submission renders", lambda r, k=reads: r[k]["cameras_rendered_per_submission"]),
            ("  render of one submission [ms]", lambda r, k=reads: r[k]["render_batch_ms"]),
            ("  render per environment step [ms]", lambda r, k=reads: r[k]["render_ms_per_env_step"]),
            ("  environment step [ms]", lambda r, k=reads: r[k]["env_step_ms"]),
        ]
    first = next(iter(results.values()))
    for camera in first["cameras"]:
        rows += [
            (f"{camera} in the submissions requested for other cameras (lazy reads)", None),
            ("  pose lag to its own read [cm]", lambda r, c=camera: mean(r, c, "pose_distance_m", 100.0)),
            ("  pose lag to its own read [deg]", lambda r, c=camera: mean(r, c, "pose_angle_deg")),
            ("  image difference to its own read [0-255]", lambda r, c=camera: mean(r, c, "image_difference")),
        ]
    lines = [
        f"{first['num_envs']} environments with {first['tasks']} tasks, cameras {', '.join(first['cameras'])},"
        f" {first['gpu']}, ovrtx {first['ovrtx']}, ovstage {first['ovstage']}",
        f"{'':52}" + "".join(f"{scene:>12}" for scene in results),
    ]
    for label, value in rows:
        if value is None:
            lines += ["", label]
            continue
        values = [value(result) for result in results.values()]
        lines.append(f"{label:52}" + "".join(f"{'-' if v is None else format(v, '.2f'):>12}" for v in values))
    for scene, result in results.items():
        lines += ["", f"Native submissions of one environment step with lazy reads, {scene}:"]
        for index, call in enumerate(result["trace"][-1]["submissions"], start=1):
            lines.append(
                f"  {index}: requested for {', '.join(call['requested'])}; rendered {', '.join(call['rendered'])};"
                f" camera poses written before it: {', '.join(call['poses_written_before']) or 'none'}"
            )
    return "\n".join(lines)


def measure() -> None:
    """Measure the scene ``--scene`` and write its ``result.json``."""
    os.environ["ISAAC_LAB_OVRTX_USE_OVSTAGE"] = SCENES[args.scene]
    output.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    # Cameras share one renderer while their renderer configurations are equal.
    for camera in vars(env_cfg.scene).values():
        if isinstance(camera, CameraCfg):
            camera.renderer_cfg.log_file_path = str(output / "ovrtx_renderer.log")
    probe = Probe()
    with launch_simulation(env_cfg, args):
        import ovrtx
        import ovstage

        probe.install()
        # The observation manager reads every camera once while the environment is created.
        started = time.perf_counter()
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        wp.synchronize()
        creation_seconds = time.perf_counter() - started
        taken = probe.take()
        cameras = {name: sensor for name, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)}
        renderer = next(iter(cameras.values()))._renderer
        if any(camera._renderer is not renderer for camera in cameras.values()):
            raise RuntimeError("The cameras do not share one renderer.")
        if renderer._use_ovstage != (args.scene == "ovstage"):
            raise RuntimeError(f"The renderer does not use the scene {args.scene!r}.")
        result = {
            "scene": args.scene,
            "num_envs": env.num_envs,
            "gpu": torch.cuda.get_device_name(env.device),
            "ovrtx": ovrtx.__version__,
            "ovstage": ovstage.__version__,
            "cameras": {
                name: {"width": camera.cfg.width, "height": camera.cfg.height, "data_types": camera.cfg.data_types}
                for name, camera in cameras.items()
            },
            "initialization": {
                "environment_creation_s": creation_seconds,
                "prepare_stage_s": sum(taken["seconds"]["prepare_stage"]),
                # The native scene initializes its transforms in a first submission that advances no render time.
                "submissions_s": sum(call["seconds"] for call in taken["submissions"]),
                "submissions": [
                    {"delta_time": call["delta_time"], "seconds": call["seconds"]} for call in taken["submissions"]
                ],
            },
        }
        print(f"[{args.scene}] environment created in {creation_seconds:.1f} s", flush=True)

        env_ids = torch.arange(env.num_envs, device=env.device)
        env.task_manager.set_task(env_ids, env_ids % len(env.cfg.tasks))
        started = time.perf_counter()
        obs, _ = env.reset()
        wp.synchronize()
        result["task_assignment_s"] = time.perf_counter() - started
        result["tasks"] = len(set(env.task_manager.task_ids.tolist()))
        # The environment's gripper action closes below zero.
        hold = torch.cat((obs["policy"]["arm_joint_pos"], 0.5 - obs["policy"]["gripper_pos"].round()), dim=1)
        actions = wrist_motion(hold)

        result["batched"] = time_steps(env, probe, actions, args.steps)
        # Isaac Lab's lazy read renders the camera that is read. This branch reads all due cameras with it.
        Camera._update_outdated_buffers = SensorBase._update_outdated_buffers
        result["lazy"] = time_steps(env, probe, actions, args.steps)
        result["trace"] = trace_lazy_reads(env, probe, cameras, actions)
        probe.submit_requested_only = True
        result["requested_only"] = time_steps(env, probe, actions, REQUESTED_ONLY_STEPS)

        result["physics_faults"] = int(env.termination_manager.get_term("solver_fault").sum())
        free, total = torch.cuda.mem_get_info(env.device)
        result["gpu_memory_used_mib"] = round((total - free) / 2**20)
        result["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        (output / "result.json").write_text(json.dumps(result, indent=1))
        print(report({args.scene: result}), flush=True)
        env.close()


def compare() -> None:
    """Measure both scenes, one process each, and write ``report.txt``."""

    def run(scene: str, directory: Path, *options: str) -> None:
        """Measure a scene in a process of its own, which gets this process's options."""
        directory.mkdir(parents=True)
        command = [sys.executable, __file__, *sys.argv[1:], "--scene", scene, "--output", str(directory), *options]
        with (directory / "run.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)

    print("[warm-up] creating a small environment, which fills the caches", flush=True)
    run("ovstage", output / "warmup", "--num_envs", "2", "--steps", "1")
    results = {}
    for scene in SCENES:
        print(f"[{scene}] measuring, log in {output / scene / 'run.log'}", flush=True)
        run(scene, output / scene)
        if (output / scene / "result.json").exists():
            results[scene] = json.loads((output / scene / "result.json").read_text())
        else:
            print(f"[{scene}] failed", flush=True)
    if results:
        (output / "report.txt").write_text(report(results) + "\n")
        print(report(results), flush=True)
    print(f"Results in {output}", flush=True)
    if len(results) < len(SCENES):
        sys.exit(1)


if __name__ == "__main__":
    if args.scene is None:
        compare()
    else:
        measure()
