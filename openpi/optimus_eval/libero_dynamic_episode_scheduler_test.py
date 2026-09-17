import argparse
from pathlib import Path

from optimus_eval.libero_dynamic_episode_scheduler import _absolute_executable_path, _client_command, _partition


def test_partition_is_disjoint_and_complete() -> None:
    shards = _partition(100, 50, 4)
    flattened = [episode for shard in shards for episode in shard]
    assert len(shards) == 4
    assert len(flattened) == len(set(flattened)) == 50
    assert set(flattened) == set(range(100, 150))
    assert sorted(map(len, shards)) == [12, 12, 13, 13]


def test_absolute_executable_path_preserves_venv_launcher(tmp_path: Path) -> None:
    system_python = tmp_path / "system-python"
    system_python.touch()
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(system_python)

    assert _absolute_executable_path(venv_python) == venv_python


def test_client_command_propagates_smoke_step_limit(tmp_path: Path) -> None:
    args = argparse.Namespace(
        libero_python=tmp_path / "python",
        openpi_root=tmp_path / "openpi",
        run_root=tmp_path / "run",
        host="127.0.0.1",
        port=8200,
        episodes_per_task=1,
        episode_start=0,
        replan_steps=10,
        num_steps_wait=10,
        resize_size=224,
        seed=7,
        environment_id_prefix="smoke",
        max_env_steps=20,
        save_videos=False,
        episode_data_root=None,
    )

    command = _client_command(args, task_id=0, shard_id=0, episodes=(0,))

    index = command.index("--args.max-env-steps")
    assert command[index + 1] == "20"
    assert command[command.index("--args.task-ids-csv") + 1] == "0"
