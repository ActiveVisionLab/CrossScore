import argparse
import os
import json
from pathlib import Path
from pprint import pprint
import numpy as np


def split_list_by_ratio(list_input, ratio_dict):
    # Check if the sum of ratios is close to 1
    if not 0.999 < sum(ratio_dict.values()) < 1.001:
        raise ValueError("The sum of the ratios must be close to 1")

    total_length = len(list_input)
    lengths = {k: int(v * total_length) for k, v in ratio_dict.items()}

    # Adjust the last split to include any rounding difference
    last_split_name = list(ratio_dict.keys())[-1]
    lengths[last_split_name] = total_length - sum(lengths.values()) + lengths[last_split_name]

    # Split the list
    split_lists = {}
    start = 0
    for split_name, length in lengths.items():
        split_lists[split_name] = list_input[start : start + length]
        start += length

    split_lists = {k: v.tolist() for k, v in split_lists.items()}
    return split_lists


if __name__ == "__main__":
    np.random.seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="~/projects/mview/storage/scratch_dataset/gaussian-splatting-processed/RealEstate200/res_540",
    )
    parser.add_argument("--min_seq_len", type=int, default=2, help="keep seq with len(imgs) >= 2")
    parser.add_argument("--min_psnr", type=float, default=10.0, help="keep seq with psnr >= 10.0")
    parser.add_argument(
        "--split_ratio",
        nargs="+",
        type=float,
        default=[0.8, 0.1, 0.1],
        help="train/val/test split ratio",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path).expanduser()
    log_files = sorted([f for f in os.listdir(data_path) if f.endswith(".log")])

    # get low psnr scenes from log files
    scene_all = []
    scene_low_psnr = {}
    for log_f in log_files:
        with open(data_path / log_f, "r") as f:
            lines = f.readlines()
            for line in lines:
                # assume scene name printed a few lines before PSNR
                if "Output folder" in line:
                    scene_name = line.split("Output folder: ")[1].split("/")[-1]
                    scene_name = scene_name.removesuffix("\n")
                elif "[ITER 7000] Evaluating train" in line:
                    psnr = line.split("PSNR ")[1]
                    psnr = psnr.removesuffix("\n")
                    psnr = float(psnr)

                    scene_all.append(scene_name)
                    if psnr < args.min_psnr:
                        scene_low_psnr[scene_name] = psnr
                else:
                    pass

    # get low seq length scenes from data folders
    gaussian_splits = ["train", "test"]
    scene_low_length = {}
    for scene_name in scene_all:
        for gs_split in gaussian_splits:
            tmp_dir = data_path / scene_name / gs_split / "ours_1000" / "gt"
            num_img = len(os.listdir(tmp_dir))
            if num_img < args.min_seq_len:
                scene_low_length[scene_name] = num_img

    num_scene_total_after_gaussian = len(scene_all)
    num_scene_low_psnr = len(scene_low_psnr)
    num_scene_low_length = len(scene_low_length)

    # filter out low psnr scenes
    scene_all = [s for s in scene_all if s not in scene_low_psnr.keys()]
    num_scene_total_filtered_low_psnr = len(scene_all)

    # filter out low seq length scenes
    scene_all = [s for s in scene_all if s not in scene_low_length.keys()]
    num_scene_total_filtered_low_length = len(scene_all)

    # split train/val/test
    scene_all = np.random.permutation(scene_all)
    num_scene_after_all_filtering = len(scene_all)
    ratio = {
        "train": args.split_ratio[0],
        "val": args.split_ratio[1],
        "test": args.split_ratio[2],
    }
    scene_split_info = split_list_by_ratio(scene_all, ratio)
    num_scene_train = len(scene_split_info["train"])
    num_scene_val = len(scene_split_info["val"])
    num_scene_test = len(scene_split_info["test"])
    num_scene_after_split = sum([len(v) for v in scene_split_info.values()])
    assert num_scene_after_split == num_scene_after_all_filtering

    # save to json
    stats = {
        "min_psnr": args.min_psnr,
        "min_seq_len": args.min_seq_len,
        "split_ratio": args.split_ratio,
        "num_scene_total_after_gaussian": num_scene_total_after_gaussian,
        "num_scene_low_psnr": num_scene_low_psnr,
        "num_scene_low_length": num_scene_low_length,
        "num_scene_total_filtered_low_psnr": num_scene_total_filtered_low_psnr,
        "num_scene_total_filtered_low_length": num_scene_total_filtered_low_length,
        "num_scene_after_all_filtering": num_scene_after_all_filtering,
        "num_scene_train": num_scene_train,
        "num_scene_val": num_scene_val,
        "num_scene_test": num_scene_test,
        "num_scene_after_split": num_scene_after_split,
    }

    pprint(stats, sort_dicts=False)
    out_dict = {"stats": stats, **scene_split_info}

    with open(data_path / "split.json", "w") as f:
        json.dump(out_dict, f, indent=2)
