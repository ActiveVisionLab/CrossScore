"""Command-line interface for CrossScore."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="CrossScore: Multi-View Image Quality Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  crossscore --query-dir path/to/queries --reference-dir path/to/references
  crossscore --query-dir renders/ --reference-dir gt/ --metric-type mae --batch-size 4
  crossscore --query-dir renders/ --reference-dir gt/ --cpu
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
        "--device",
        default=None,
        help="Device string, e.g. 'cuda', 'cuda:0', 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (no GPU)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for results (default: ./crossscore_output)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write score map images to disk",
    )

    args = parser.parse_args()

    from crossscore.api import score

    device = "cpu" if args.cpu else args.device

    results = score(
        query_dir=args.query_dir,
        reference_dir=args.reference_dir,
        ckpt_path=args.ckpt_path,
        metric_type=args.metric_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resize_short_side=args.resize_short_side,
        device=device,
        out_dir=args.out_dir,
        write_score_maps=not args.no_write,
    )

    n_images = len(results["scores"])
    print(f"\nCrossScore completed: {n_images} images scored")
    if results["scores"]:
        mean_score = sum(results["scores"]) / len(results["scores"])
        print(f"Mean score: {mean_score:.4f}")
        for i, s in enumerate(results["scores"]):
            print(f"  Image {i}: {s:.4f}")
    if "out_dir" in results:
        print(f"Score maps written to: {results['out_dir']}")


if __name__ == "__main__":
    main()
