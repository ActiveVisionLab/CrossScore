"""Command-line interface for CrossScore."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="CrossScore: Multi-View Image Quality Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  crossscore --query-dir path/to/queries --reference-dir path/to/references
  crossscore --query-dir renders/ --reference-dir gt/ --metric-type mae --batch-size 4
  crossscore --query-dir renders/ --reference-dir gt/ --ckpt-path my_model.ckpt
""",
    )
    parser.add_argument(
        "--query-dir", required=True, help="Directory containing query images"
    )
    parser.add_argument(
        "--reference-dir", required=True, help="Directory containing reference images"
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Path to model checkpoint (auto-downloads if not provided)",
    )
    parser.add_argument(
        "--metric-type",
        default="ssim",
        choices=["ssim", "mae", "mse"],
        help="Metric type to predict (default: ssim)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Data loading workers (default: 4)"
    )
    parser.add_argument(
        "--resize-short-side",
        type=int,
        default=518,
        help="Resize short side to this value, -1 to disable (default: 518)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=None,
        help="GPU device indices (default: [0])",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for results (default: auto-generated)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write output files to disk",
    )

    args = parser.parse_args()

    from crossscore.api import score

    results = score(
        query_dir=args.query_dir,
        reference_dir=args.reference_dir,
        ckpt_path=args.ckpt_path,
        metric_type=args.metric_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resize_short_side=args.resize_short_side,
        devices=args.devices,
        out_dir=args.out_dir,
        write_outputs=not args.no_write,
    )

    n_maps = sum(s.shape[0] for s in results["score_maps"]) if results["score_maps"] else 0
    print(f"\nCrossScore completed: {n_maps} score maps generated")
    if "out_dir" in results and results["out_dir"]:
        print(f"Results written to: {results['out_dir']}")


if __name__ == "__main__":
    main()
