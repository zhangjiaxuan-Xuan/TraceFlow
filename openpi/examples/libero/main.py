from __future__ import annotations

import collections
import dataclasses
import fcntl
import json
import logging
import math
import os
import pathlib
import tempfile
import time
from typing import Any

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.video_utils import VideoWriter
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LOG_TS_FMT = "%Y-%m-%d %H:%M:%S"


def _now_str() -> str:
    return time.strftime(LOG_TS_FMT, time.localtime())


def safe_append(log_file: str, record: dict[str, Any]) -> None:
    path = pathlib.Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_completed_episodes(log_file: str, task_suite: str) -> set[tuple[int, int]]:
    """Load successfully evaluated task/episode keys for resume."""
    if not log_file:
        return set()
    path = pathlib.Path(log_file)
    if not path.is_file():
        raise FileNotFoundError(f"Resume log not found: {path}")
    completed = set()
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in resume log {path}:{line_number}") from exc
            if record.get("event") != "episode_result" or record.get("task_suite") != task_suite:
                continue
            if record.get("error"):
                continue
            completed.add((int(record["task_id"]), int(record["episode_idx"])))
    return completed


def archive_interrupted_path(path: pathlib.Path) -> pathlib.Path | None:
    """Preserve an incomplete artifact before rerunning its episode."""
    if not path.exists():
        return None
    suffix = f".interrupted.{int(time.time())}.{os.getpid()}"
    archived = path.with_name(path.name + suffix)
    os.replace(path, archived)
    return archived


def atomic_savez_compressed(path: pathlib.Path, **arrays: Any) -> None:
    """Write a pickle-free NPZ without exposing a partially written final file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing observation dump: {path}")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
            np.savez_compressed(tmp, **arrays)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 10
    log_file: str = "./libero_results.jsonl"
    save_videos: bool = False
    video_root: str = "./videos"
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    max_env_steps: int = 0
    num_trials_per_task: int = 50
    episode_start: int = 0
    seed: int = 7
    dump_observation_npz: str = ""
    dump_observation_dir: str = ""
    task_ids_csv: str = ""
    episode_indices_csv: str = ""
    environment_id: str = "env-0"
    resume_log_file: str = ""
    resume_worker_id: int = 0
    resume_num_workers: int = 1
    episode_data_root: str = ""
    episode_data_mode: str = "all"
    trajectory_image_size: int = 128
    fail_on_episode_error: bool = False


class EpisodeTrajectoryWriter:
    """Stream one rollout into an atomic, LIBERO-compatible HDF5 episode."""

    def __init__(
        self,
        *,
        root: pathlib.Path,
        suite: str,
        task_id: int,
        episode_idx: int,
        prompt: str,
        init_state: np.ndarray,
        model_xml: str,
        bddl_file_name: str,
        bddl_file_content: str,
        seed: int,
        image_size: int,
        resume: bool = False,
    ) -> None:
        try:
            import h5py
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Saving episode trajectories requires h5py in the LIBERO environment"
            ) from exc
        self.root = root
        self.suite = suite
        self.task_id = int(task_id)
        self.episode_idx = int(episode_idx)
        self.image_size = int(image_size)
        pending = root / suite / f"task_{task_id:03d}" / "pending"
        pending.mkdir(parents=True, exist_ok=True)
        self.tmp_path = pending / f".episode_{episode_idx:03d}.hdf5.tmp"
        if self.tmp_path.exists():
            if not resume:
                raise FileExistsError(f"Refusing to overwrite pending trajectory: {self.tmp_path}")
            archived = archive_interrupted_path(self.tmp_path)
            logging.warning("Archived interrupted trajectory: %s", archived)
        self.file = h5py.File(self.tmp_path, "w")
        self.data = self.file.create_group("data")
        self.demo = self.data.create_group("demo_0")
        self.obs = self.demo.create_group("obs")
        self._datasets: dict[str, Any] = {}
        self.data.attrs["format"] = "libero-eval-trajectory-v1"
        self.data.attrs["num_demos"] = 1
        self.data.attrs["task_suite"] = suite
        self.data.attrs["task_id"] = self.task_id
        self.data.attrs["bddl_file_name"] = bddl_file_name
        self.data.attrs["bddl_file_content"] = bddl_file_content
        self.demo.attrs["task_suite"] = suite
        self.demo.attrs["task_id"] = self.task_id
        self.demo.attrs["episode_idx"] = self.episode_idx
        self.demo.attrs["language_instruction"] = prompt
        self.demo.attrs["model_file"] = model_xml
        self.demo.attrs["init_state"] = np.asarray(init_state)
        self.demo.attrs["seed"] = int(seed)

    def _append(self, key: str, value: Any, *, observation: bool = False, image: bool = False) -> None:
        parent = self.obs if observation else self.demo
        full_key = f"obs/{key}" if observation else key
        array = np.asarray(value)
        dataset = self._datasets.get(full_key)
        if dataset is None:
            kwargs: dict[str, Any] = {
                "shape": (0, *array.shape),
                "maxshape": (None, *array.shape),
                "dtype": array.dtype,
                "chunks": (1, *array.shape) if array.shape else (256,),
            }
            if image:
                kwargs.update(compression="gzip", compression_opts=1, shuffle=True)
            dataset = parent.create_dataset(key, **kwargs)
            self._datasets[full_key] = dataset
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = array

    def append(
        self,
        *,
        state: np.ndarray,
        action: np.ndarray,
        observation: dict[str, Any],
        reward: float,
        terminal: bool,
        env_step: int,
        policy_call_idx: int,
        chunk_offset: int,
        robot_state: np.ndarray,
    ) -> None:
        agentview = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                np.asarray(observation["agentview_image"]), self.image_size, self.image_size
            )
        )
        eye_in_hand = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                np.asarray(observation["robot0_eye_in_hand_image"]), self.image_size, self.image_size
            )
        )
        ee_state = np.concatenate(
            (observation["robot0_eef_pos"], _quat2axisangle(np.asarray(observation["robot0_eef_quat"]).copy()))
        ).astype(np.float32)
        self._append("states", np.asarray(state, dtype=np.float64))
        self._append("actions", np.asarray(action, dtype=np.float32))
        self._append("rewards", np.asarray(reward, dtype=np.float32))
        self._append("dones", np.asarray(terminal, dtype=np.uint8))
        self._append("terminals", np.asarray(terminal, dtype=np.uint8))
        self._append("timeouts", np.asarray(0, dtype=np.uint8))
        self._append("env_steps", np.asarray(env_step, dtype=np.int32))
        self._append("policy_call_idx", np.asarray(policy_call_idx, dtype=np.int32))
        self._append("action_chunk_offset", np.asarray(chunk_offset, dtype=np.int16))
        self._append("robot_states", np.asarray(robot_state, dtype=np.float32))
        self._append("agentview_rgb", agentview, observation=True, image=True)
        self._append("eye_in_hand_rgb", eye_in_hand, observation=True, image=True)
        self._append("gripper_states", np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32), observation=True)
        self._append("joint_states", np.asarray(observation["robot0_joint_pos"], dtype=np.float32), observation=True)
        self._append("ee_states", ee_state, observation=True)
        self._append("ee_pos", ee_state[:3], observation=True)
        self._append("ee_ori", ee_state[3:], observation=True)

    def finalize(self, *, success: bool, error: str, keep: bool) -> pathlib.Path | None:
        num_samples = int(self._datasets.get("actions").shape[0]) if "actions" in self._datasets else 0
        if num_samples:
            self._datasets["dones"][-1] = 1
            if not success and not error:
                self._datasets["timeouts"][-1] = 1
        self.demo.attrs["num_samples"] = num_samples
        self.demo.attrs["success"] = bool(success)
        self.demo.attrs["error"] = error
        self.data.attrs["total"] = num_samples
        self.file.flush()
        self.file.close()
        if not keep:
            self.tmp_path.unlink(missing_ok=True)
            return None

        label = "success" if success else "failed"
        marker = "SUCCESS" if success else "FAILED"
        final_dir = self.root / self.suite / f"task_{self.task_id:03d}" / label
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / f"{marker}_episode_{self.episode_idx:03d}.hdf5"
        if final_path.exists():
            raise FileExistsError(f"Refusing to overwrite trajectory: {final_path}")
        with self.tmp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(self.tmp_path, final_path)
        return final_path


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)
    if args.episode_start < 0 or args.num_trials_per_task <= 0:
        raise ValueError("episode_start must be non-negative and num_trials_per_task must be positive")
    if args.episode_data_mode not in {"all", "failures", "successes", "none"}:
        raise ValueError("episode_data_mode must be one of: all, failures, successes, none")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    max_steps = args.max_env_steps if args.max_env_steps > 0 else _max_steps_for_suite(args.task_suite_name)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    safe_append(
        args.log_file,
        {
            "ts": _now_str(),
            "event": "process_start",
            "pid": os.getpid(),
            "task_suite": args.task_suite_name,
            "port": args.port,
            "seed": args.seed,
            "replan_steps": args.replan_steps,
        },
    )

    if args.task_ids_csv:
        task_ids = [int(x.strip()) for x in args.task_ids_csv.split(",") if x.strip()]
    else:
        task_ids = list(range(num_tasks_in_suite))
    completed_episodes = load_completed_episodes(args.resume_log_file, args.task_suite_name)
    if completed_episodes:
        logging.info("Resume enabled: loaded %d completed episodes from %s", len(completed_episodes), args.resume_log_file)
    if args.resume_num_workers <= 0 or not 0 <= args.resume_worker_id < args.resume_num_workers:
        raise ValueError("resume_worker_id must be in [0, resume_num_workers)")

    total_episodes = 0
    total_successes = 0
    resume_shard_offset = 0
    for task_id in tqdm.tqdm(task_ids):
        if task_id < 0 or task_id >= num_tasks_in_suite:
            raise ValueError(f"task_id {task_id} is out of range for {args.task_suite_name}: 0..{num_tasks_in_suite - 1}")
        episode_indices = (
            [int(x.strip()) for x in args.episode_indices_csv.split(",") if x.strip()]
            if args.episode_indices_csv
            else list(range(args.episode_start, args.episode_start + args.num_trials_per_task))
        )
        for episode_idx in episode_indices:
            if episode_idx < args.episode_start or episode_idx >= args.episode_start + args.num_trials_per_task:
                raise ValueError(
                    f"episode_idx {episode_idx} is outside "
                    f"{args.episode_start}..{args.episode_start + args.num_trials_per_task - 1}"
                )
        episode_indices = [
            episode_idx for episode_idx in episode_indices if (int(task_id), int(episode_idx)) not in completed_episodes
        ]
        if args.resume_log_file:
            missing_count = len(episode_indices)
            episode_indices = [
                episode_idx
                for position, episode_idx in enumerate(episode_indices)
                if (resume_shard_offset + position) % args.resume_num_workers == args.resume_worker_id
            ]
            resume_shard_offset += missing_count
        if not episode_indices:
            logging.info("Resume: task %d has no missing episodes for this worker", task_id)
            continue
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if len(initial_states) == 0:
            raise ValueError(f"Task {task_id} has no init states")
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        _, task_bddl_path = _resolve_libero_task_bddl_paths(task)
        task_bddl_content = task_bddl_path.read_text(encoding="utf-8")
        task_video_root = pathlib.Path(args.video_root) / args.task_suite_name / f"task_{task_id:03d}"

        task_episodes = 0
        task_successes = 0
        for episode_idx in tqdm.tqdm(episode_indices):
            init_state_idx = episode_idx % len(initial_states)
            logging.info("Task: %s", task_description)
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[init_state_idx])
            t = 0
            done = False
            episode_reset_pending = True
            policy_call_idx = 0
            episode_error = ""
            pending_video_path = task_video_root / "pending" / f"episode_{episode_idx:03d}"
            if args.resume_log_file:
                archived = archive_interrupted_path(pending_video_path)
                if archived is not None:
                    logging.warning("Archived interrupted video: %s", archived)
            episode_video_path: pathlib.Path | None = None
            trajectory_path: pathlib.Path | None = None
            trajectory_writer = None
            if args.episode_data_root and args.episode_data_mode != "none":
                trajectory_writer = EpisodeTrajectoryWriter(
                    root=pathlib.Path(args.episode_data_root),
                    suite=args.task_suite_name,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    prompt=str(task_description),
                    init_state=np.asarray(initial_states[init_state_idx]),
                    model_xml=env.sim.model.get_xml(),
                    bddl_file_name=str(task_bddl_path),
                    bddl_file_content=task_bddl_content,
                    seed=args.seed,
                    image_size=args.trajectory_image_size,
                    resume=bool(args.resume_log_file),
                )

            with VideoWriter(str(pending_video_path), save_video=args.save_videos) as video_writer:
                if args.save_videos:
                    video_writer.append_obs(obs, done=False)

                while t < max_steps + args.num_steps_wait:
                    try:
                        if t < args.num_steps_wait:
                            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                            if args.save_videos:
                                video_writer.append_obs(obs, done)
                            t += 1
                            if done:
                                task_successes += 1
                                total_successes += 1
                                break
                            continue

                        image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        image = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(image, args.resize_size, args.resize_size)
                        )
                        wrist_image = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_image, args.resize_size, args.resize_size)
                        )

                        if not action_plan:
                            executed_steps = max(0, t - args.num_steps_wait)
                            progress = float(np.clip(executed_steps / max_steps, 0.0, 1.0))
                            element = {
                                "observation/image": image,
                                "observation/wrist_image": wrist_image,
                                "observation/state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        _quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],
                                    )
                                ),
                                "prompt": str(task_description),
                                "progress": progress,
                                "episode_reset": episode_reset_pending,
                                "trace_context": {
                                    "environment_id": args.environment_id,
                                    "task_suite": args.task_suite_name,
                                    "task_id": int(task_id),
                                    "episode_idx": int(episode_idx),
                                    "policy_call_idx": int(policy_call_idx),
                                    "env_step": int(t),
                                },
                            }
                            dump_path = None
                            if args.dump_observation_dir:
                                dump_path = (
                                    pathlib.Path(args.dump_observation_dir)
                                    / args.task_suite_name
                                    / f"task_{task_id:03d}"
                                    / f"episode_{episode_idx:03d}"
                                    / f"call_{policy_call_idx:04d}_step_{t:04d}.npz"
                                )
                            elif args.dump_observation_npz and not pathlib.Path(args.dump_observation_npz).exists():
                                dump_path = pathlib.Path(args.dump_observation_npz)

                            if dump_path is not None:
                                atomic_savez_compressed(
                                    dump_path,
                                    observation_image=element["observation/image"],
                                    observation_wrist_image=element["observation/wrist_image"],
                                    observation_state=element["observation/state"],
                                    prompt=np.asarray(element["prompt"]),
                                    progress=np.asarray(progress, dtype=np.float32),
                                    task_suite_name=np.asarray(args.task_suite_name),
                                    task_id=np.asarray(task_id, dtype=np.int64),
                                    episode_idx=np.asarray(episode_idx, dtype=np.int64),
                                    step=np.asarray(t, dtype=np.int64),
                                    policy_call_idx=np.asarray(policy_call_idx, dtype=np.int64),
                                )
                                safe_append(
                                    args.log_file,
                                    {
                                        "ts": _now_str(),
                                        "event": "observation_dump",
                                        "path": str(dump_path),
                                        "task_suite": args.task_suite_name,
                                        "task_id": int(task_id),
                                        "episode_idx": int(episode_idx),
                                        "step": int(t),
                                        "policy_call_idx": int(policy_call_idx),
                                        "progress": float(progress),
                                    },
                                )
                            action_chunk = client.infer(element)["actions"]
                            source_call_idx = policy_call_idx
                            policy_call_idx += 1
                            episode_reset_pending = False
                            if len(action_chunk) < args.replan_steps:
                                raise RuntimeError(
                                    f"Policy returned {len(action_chunk)} actions, "
                                    f"but replan_steps is {args.replan_steps}."
                                )
                            action_plan.extend(
                                (action, source_call_idx, chunk_offset)
                                for chunk_offset, action in enumerate(action_chunk[: args.replan_steps])
                            )

                        action, action_call_idx, chunk_offset = action_plan.popleft()
                        state_before_action = env.get_sim_state().copy()
                        obs, reward, done, _ = env.step(action.tolist())
                        if trajectory_writer is not None:
                            trajectory_writer.append(
                                state=state_before_action,
                                action=action,
                                observation=obs,
                                reward=reward,
                                terminal=done,
                                env_step=t,
                                policy_call_idx=action_call_idx,
                                chunk_offset=chunk_offset,
                                robot_state=env.env.get_robot_state_vector(obs),
                            )
                        if args.save_videos:
                            video_writer.append_obs(obs, done)
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1
                    except Exception as exc:
                        episode_error = f"{type(exc).__name__}: {exc}"
                        logging.exception("Episode failed: %s", exc)
                        break

            label = "success" if done else "failed"
            marker = "SUCCESS" if done else "FAILED"
            if args.save_videos:
                final_video_dir = task_video_root / label / f"{marker}_episode_{episode_idx:03d}"
                final_video_dir.parent.mkdir(parents=True, exist_ok=True)
                if final_video_dir.exists():
                    raise FileExistsError(f"Refusing to overwrite video directory: {final_video_dir}")
                os.replace(pending_video_path, final_video_dir)
                episode_video_path = final_video_dir / "video.mp4"

            if trajectory_writer is not None:
                keep_trajectory = (
                    args.episode_data_mode == "all"
                    or (args.episode_data_mode == "failures" and not done)
                    or (args.episode_data_mode == "successes" and done)
                )
                trajectory_path = trajectory_writer.finalize(
                    success=bool(done), error=episode_error, keep=keep_trajectory
                )

            task_episodes += 1
            total_episodes += 1
            safe_append(
                args.log_file,
                {
                    "ts": _now_str(),
                    "event": "episode_result",
                    "task_suite": args.task_suite_name,
                    "task_id": int(task_id),
                    "episode_idx": int(episode_idx),
                    "seed": int(args.seed),
                    "success": bool(done),
                    "env_steps": int(t),
                    "policy_calls": int(policy_call_idx),
                    "error": episode_error,
                    "video_path": str(episode_video_path) if episode_video_path is not None else "",
                    "trajectory_path": str(trajectory_path) if trajectory_path is not None else "",
                },
            )
            if episode_error and args.fail_on_episode_error:
                raise RuntimeError(
                    f"Aborting worker after policy/environment error for "
                    f"{args.task_suite_name} task={task_id} episode={episode_idx}: {episode_error}"
                )
            logging.info("Success: %s", done)
            logging.info("Episodes: %d Successes: %d Rate: %.1f%%", total_episodes, total_successes, 100.0 * total_successes / total_episodes)
            if args.save_videos:
                safe_append(
                    args.log_file,
                    {
                        "ts": _now_str(),
                        "event": "episode_video",
                        "task_suite": args.task_suite_name,
                        "task_id": int(task_id),
                        "episode_idx": int(episode_idx),
                        "success": bool(done),
                        "video_path": str(episode_video_path),
                    },
                )

        logging.info("Task success rate: %.4f", float(task_successes) / float(task_episodes))
        logging.info("Current total success rate: %.4f", float(total_successes) / float(total_episodes))
        env.close()

    safe_append(
        args.log_file,
        {
            "ts": _now_str(),
            "event": "run_summary",
            "pid": os.getpid(),
            "task_suite": args.task_suite_name,
            "port": args.port,
            "total_episodes": int(total_episodes),
            "total_successes": int(total_successes),
            "total_success_rate": float(total_successes / total_episodes) if total_episodes else 0.0,
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
        },
    )


def _max_steps_for_suite(task_suite_name: str) -> int:
    match task_suite_name:
        case "libero_spatial":
            return 220
        case "libero_object":
            return 280
        case "libero_goal":
            return 300
        case "libero_10":
            return 520
        case "libero_90":
            return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _resolve_libero_task_bddl_paths(task) -> tuple[pathlib.Path, pathlib.Path]:
    encoded_path = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    encoded_text = str(encoded_path)
    if "_view_" in encoded_text and "_initstate_" in encoded_text:
        source_path = pathlib.Path(encoded_text.split("_view_", 1)[0] + ".bddl")
    else:
        source_path = encoded_path
    return encoded_path, source_path


def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file, _ = _resolve_libero_task_bddl_paths(task)
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
