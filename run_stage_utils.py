import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence


RUN_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"


def collect_scene_images(
    data_root: Path,
    scene_names: Optional[Sequence[str]] = None,
    prepare_images: bool = True,
) -> List[Path]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    scene_dirs = sorted(
        path
        for path in data_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not scene_dirs:
        raise RuntimeError(f"No scene directories found under: {data_root}")

    if scene_names:
        scenes_by_name = {scene_dir.name: scene_dir for scene_dir in scene_dirs}
        requested_names = list(dict.fromkeys(scene_names))
        missing_names = [
            scene_name
            for scene_name in requested_names
            if scene_name not in scenes_by_name
        ]
        if missing_names:
            raise FileNotFoundError(
                f"Scene directories not found under {data_root}: "
                f"{', '.join(missing_names)}"
            )
        scene_dirs = [scenes_by_name[scene_name] for scene_name in requested_names]

    scene_images = []
    for scene_dir in scene_dirs:
        image_path = scene_dir / f"{scene_dir.name}_rgba.png"
        if not image_path.is_file():
            source_path = scene_dir / "rgba.png"
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Missing both scene image {image_path} and fallback {source_path}"
                )
            if prepare_images:
                shutil.copyfile(source_path, image_path)
                print(f"Created scene image: {image_path}", file=sys.stderr)
        scene_images.append(image_path)
    return scene_images


def find_latest_checkpoint(
    previous_exp_root: Path, previous_stage: str, image_name: str
) -> Path:
    stage_dir = previous_exp_root / previous_stage
    if not stage_dir.is_dir():
        raise FileNotFoundError(f"Previous stage directory does not exist: {stage_dir}")

    run_prefix = f"{image_name}@"
    candidates = []
    for run_dir in stage_dir.iterdir():
        checkpoint = run_dir / "ckpts" / "last.ckpt"
        if (
            run_dir.is_dir()
            and run_dir.name.startswith(run_prefix)
            and checkpoint.is_file()
        ):
            timestamp_text = run_dir.name[len(run_prefix) :]
            try:
                timestamp = datetime.strptime(timestamp_text, RUN_TIMESTAMP_FORMAT)
            except ValueError as error:
                raise RuntimeError(
                    f"Invalid run timestamp in directory name: {run_dir.name}"
                ) from error
            candidates.append((timestamp, checkpoint))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint matching {image_name}@*/ckpts/last.ckpt under {stage_dir}"
        )

    candidates.sort(key=lambda item: item[0])
    checkpoint = candidates[-1][1]
    if len(candidates) > 1:
        print(
            f"Found {len(candidates)} checkpoints for {image_name}; "
            f"using latest: {checkpoint}",
            file=sys.stderr,
        )
    return checkpoint


def run_command(command: Sequence[str], dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)
