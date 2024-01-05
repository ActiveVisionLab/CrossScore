from argparse import ArgumentParser
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))
from utils.io.score_summariser import SummaryWriterGroundTruth


def parse_args():
    parser = ArgumentParser(description="Summarise the ground truth results.")
    parser.add_argument(
        "--dir_in",
        type=str,
        default="datadir/processed_training_ready/gaussian/map-free-reloc/res_540",
        help="The ground truth data dir that contains scene dirs.",
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        default="~/projects/mview/storage/scratch_dataset/score_summary",
        help="The output directory to save the summarised results.",
    )
    parser.add_argument(
        "--fast_debug",
        type=int,
        default=-1,
        help="num batch to load for debug. Set to -1 to disable",
    )
    parser.add_argument("-n", "--num_workers", type=int, default=16)
    parser.add_argument("-f", "--force", type=eval, default=False, choices=[True, False])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summariser = SummaryWriterGroundTruth(
        dir_in=args.dir_in,
        dir_out=args.dir_out,
        num_workers=args.num_workers,
        fast_debug=args.fast_debug,
        force=args.force,
    )
    summariser.write_csv()
