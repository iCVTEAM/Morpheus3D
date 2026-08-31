import argparse
from pathlib import Path

from run_stage_utils import collect_scene_images, run_command


CONFIG_PATH = "configs/morpheus3d-coarse.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Morpheus3D coarse stage.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--exp-root", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--scene",
        dest="scenes",
        action="append",
        metavar="NAME",
        help="Run only the named scene. Repeat the option to select multiple scenes.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for image_path in collect_scene_images(
        args.data_root, args.scenes, prepare_images=not args.dry_run
    ):
        command = [
            "python",
            "launch.py",
            "--config",
            CONFIG_PATH,
            "--train",
            "--gpu",
            str(args.gpu),
            f"data.image_path={image_path}",
            f"exp_root_dir={args.exp_root}",
            "system.guidance_3d.guidance_scale=3.0",
            "system.loss.lambda_c=0.0005",
        ]
        run_command(command, args.dry_run)


if __name__ == "__main__":
    main()
