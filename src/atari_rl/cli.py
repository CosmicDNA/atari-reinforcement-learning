import argparse

from . import enjoy, record_video, train


def main():
    """Main entry point for the atari-rl command-line interface."""
    parser = argparse.ArgumentParser(
        description="A CLI for training and evaluating reinforcement learning agents on Atari games.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # --- Train Command ---
    parser_train = subparsers.add_parser("train", help="Train a new agent or resume training.")
    parser_train.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint.")
    parser_train.set_defaults(func=lambda args: train.start_training(resume=args.resume))

    # --- Enjoy Command ---
    parser_enjoy = subparsers.add_parser("enjoy", help="Watch a trained agent play.")
    parser_enjoy.add_argument("--exp-id", type=int, help="Experiment ID to enjoy (if not provided, latest is used).")
    parser_enjoy.set_defaults(func=lambda args: enjoy.watch_agent(exp_id=args.exp_id))

    # --- Record Video Command ---
    parser_record = subparsers.add_parser("record-video", help="Record a video of a trained agent.")
    parser_record.add_argument("--exp-id", type=int, help="Experiment ID to record (if not provided, latest is used).")
    parser_record.add_argument("--format", type=str, choices=["mp4", "svg", "all"], default="mp4", help="Recording format (mp4, svg, or all). Default: mp4")
    parser_record.set_defaults(func=lambda args: record_video.create_video(exp_id=args.exp_id, video_format=args.format))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()