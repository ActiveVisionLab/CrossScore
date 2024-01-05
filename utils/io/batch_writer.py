import json
from pathlib import Path
from PIL import Image
import numpy as np
from utils.io.images import metric_map_write, u8
from utils.misc.image import gray2rgb, attn2rgb, de_norm_img


def get_vrange(predict_metric_type, predict_metric_min, predict_metric_max):
    # Using uint16 when write in gray, normalise to the intrinsic range
    # based on score type, regardless of the model prediction range
    if predict_metric_type == "ssim":
        vrange_intrinsic = [-1, 1]
    elif predict_metric_type in ["mse", "mae"]:
        vrange_intrinsic = [0, 1]
    else:
        raise ValueError(f"metric_type {predict_metric_type} not supported")

    # RGB for visualization only, normalise to the model prediction range
    vrange_vis = [predict_metric_min, predict_metric_max]
    return vrange_intrinsic, vrange_vis


class BatchWriter:
    """Write batch outputs to disk."""

    def __init__(self, cfg, phase: str, img_mean_std):
        if phase not in ["test", "predict"]:
            raise ValueError(
                f"Phase {phase} not supported. Has to be a Lightening phase test/predict."
            )
        self.cfg = cfg
        self.phase = phase
        self.img_mean_std = img_mean_std

        self.out_dir = Path(self.cfg.logger[phase].out_dir)
        self.write_config = self.cfg.logger[phase].write.config
        self.write_flag = self.cfg.logger[phase].write.flag

        # overwrite the flag for attn_weights if the model does not have it
        self.write_flag.attn_weights = (
            self.write_flag.attn_weights and self.cfg.model.need_attn_weights
        )

        self.predict_metric_type = self.cfg.model.predict.metric.type
        self.predict_metric_min = self.cfg.model.predict.metric.min
        self.predict_metric_max = self.cfg.model.predict.metric.max

        self.vrange_intrinsic, self.vrange_vis = get_vrange(
            self.predict_metric_type, self.predict_metric_min, self.predict_metric_max
        )

        # prepare out_dirs, create them if required
        self.out_dir_dict = {"batch": Path(self.out_dir, "batch")}
        if self.write_flag["batch"]:
            for k in self.write_flag.keys():
                if k not in ["batch", "score_map_prediction"]:
                    if self.write_flag[k]:
                        self.out_dir_dict[k] = Path(self.out_dir_dict["batch"], k)
                        self.out_dir_dict[k].mkdir(parents=True, exist_ok=True)

    def write_out(self, batch_input, batch_output, local_rank, batch_idx):
        if self.write_flag["score_map_prediction"]:
            self._write_score_map_prediction(
                self.out_dir_dict["batch"],
                batch_input,
                batch_output,
                local_rank,
                batch_idx,
            )

        if self.write_flag["score_map_gt"]:
            self._write_score_map_gt(
                self.out_dir_dict["score_map_gt"],
                batch_input,
                local_rank,
                batch_idx,
            )

        if self.write_flag["item_path_json"]:
            self._write_item_path_json(
                self.out_dir_dict["item_path_json"],
                batch_input,
                local_rank,
                batch_idx,
            )

        if self.write_flag["image_query"]:
            self._write_query_image(
                self.out_dir_dict["image_query"],
                batch_input,
                local_rank,
                batch_idx,
            )

        if self.write_flag["image_reference"]:
            self._write_reference_image(
                self.out_dir_dict["image_reference"],
                batch_input,
                local_rank,
                batch_idx,
            )

        if self.write_flag["attn_weights"]:
            self._write_attn_weights(
                self.out_dir_dict["attn_weights"],
                batch_input,
                batch_output,
                local_rank,
                batch_idx,
                check_patch_mode="centre",
            )

    def _write_score_map_prediction(
        self, out_dir, batch_input, batch_output, local_rank, batch_idx
    ):
        query_img_paths = [
            str(Path(*Path(p).parts[-5:])).replace("/", "_").replace(".png", "")
            for p in batch_input["item_paths"]["query/img"]
        ]
        score_map_type_list = [k for k in batch_output.keys() if k.startswith("score_map")]
        for score_map_type in score_map_type_list:
            tmp_out_dir = Path(out_dir, score_map_type)
            tmp_out_dir.mkdir(parents=True, exist_ok=True)

            if len(query_img_paths) != len(batch_output[score_map_type]):
                raise ValueError("num of query images and score maps are not equal")

            for b, (query_img_p, score_map) in enumerate(
                zip(query_img_paths, batch_output[score_map_type])
            ):
                tmp_out_name = f"r{local_rank}_B{batch_idx:04}_b{b:03}_{query_img_p}.png"
                self._write_a_score_map_with_colour_mode(
                    out_path=tmp_out_dir / tmp_out_name, score_map=score_map.cpu().numpy()
                )

    def _write_score_map_gt(self, out_dir, batch_input, local_rank, batch_idx):
        query_img_paths = [
            str(Path(*Path(p).parts[-5:])).replace("/", "_").replace(".png", "")
            for p in batch_input["item_paths"]["query/img"]
        ]

        if len(query_img_paths) != len(batch_input["query/score_map"]):
            raise ValueError("num of query images and score maps are not equal")

        for b, (query_img_p, score_map) in enumerate(
            zip(query_img_paths, batch_input["query/score_map"])
        ):
            tmp_out_name = f"r{local_rank}_B{batch_idx:04}_b{b:03}_{query_img_p}.png"
            self._write_a_score_map_with_colour_mode(
                out_path=out_dir / tmp_out_name, score_map=score_map.cpu().numpy()
            )

    def _write_item_path_json(self, out_dir, batch_input, local_rank, batch_idx):
        out_path = out_dir / f"r{local_rank}_B{str(batch_idx).zfill(4)}.json"

        # get a deep copy to avoid infering other writing functions
        item_paths = batch_input["item_paths"].copy()
        for ref_type in ["reference/cross/imgs"]:
            if len(item_paths[ref_type]) > 0:
                # transpose ref paths to (N_ref, B)
                item_paths[ref_type] = np.array(item_paths[ref_type]).T.tolist()

        with open(out_path, "w") as f:
            json.dump(item_paths, f, indent=2)

    def _write_query_image(self, out_dir, batch_input, local_rank, batch_idx):
        query_img_paths = [
            str(Path(*Path(p).parts[-5:])).replace("/", "_").replace(".png", "")
            for p in batch_input["item_paths"]["query/img"]
        ]

        for b, (query_img_p, image_query) in enumerate(
            zip(query_img_paths, batch_input["query/img"])
        ):
            tmp_out_name = f"r{local_rank}_B{batch_idx:04}_b{b:03}_{query_img_p}.png"
            tmp_out_path = Path(out_dir, tmp_out_name)
            image_query = de_norm_img(image_query.permute(1, 2, 0), self.img_mean_std)
            image_query = u8(image_query.cpu().numpy())
            Image.fromarray(image_query).save(tmp_out_path)

    def _write_reference_image(self, out_dir, batch_input, local_rank, batch_idx):
        query_img_paths = [
            str(Path(*Path(p).parts[-5:])).replace("/", "_").replace(".png", "")
            for p in batch_input["item_paths"]["query/img"]
        ]
        for ref_type in ["reference/cross/imgs"]:
            if len(batch_input["item_paths"][ref_type]) > 0:
                ref_img_paths = np.array(batch_input["item_paths"][ref_type]).T  # (B, N_ref)

                # create a subfolder for each query image to store its ref images
                for b, query_img_p in enumerate(query_img_paths):
                    tmp_out_dir = Path(
                        out_dir,
                        f"r{local_rank}_B{batch_idx:04}_b{b:03}_{query_img_p}",
                        ref_type.split("/")[1],
                    )
                    tmp_out_dir.mkdir(parents=True, exist_ok=True)
                    tmp_ref_img_paths = [
                        str(Path(*Path(p).parts[-5:])).replace("/", "_").replace(".png", "")
                        for p in ref_img_paths[b]  # (N_ref, )
                    ]
                    ref_imgs = batch_input[ref_type][b]  # (N_ref, C, H, W)
                    for ref_idx, (ref_img_p, ref_img) in enumerate(
                        zip(tmp_ref_img_paths, ref_imgs)
                    ):
                        tmp_out_name = f"ref{ref_idx:02}_{ref_img_p}.png"
                        tmp_out_path = Path(tmp_out_dir, tmp_out_name)
                        ref_img = de_norm_img(ref_img.permute(1, 2, 0), self.img_mean_std)
                        ref_img = u8(ref_img.cpu().numpy())
                        Image.fromarray(ref_img).save(tmp_out_path)

    def _write_attn_weights(
        self, out_dir, batch_input, batch_output, local_rank, batch_idx, check_patch_mode
    ):
        query_img_paths = [
            str(Path(*Path(p).parts[-5:])).replace("/", "_").replace(".png", "")
            for p in batch_input["item_paths"]["query/img"]
        ]
        for ref_type in ["reference/cross/imgs"]:
            if len(batch_input["item_paths"][ref_type]) > 0:
                ref_img_paths = np.array(batch_input["item_paths"][ref_type]).T  # (B, N_ref)
                ref_type_short = ref_type.split("/")[1]

                # create a subfolder for each query image to store its ref images
                for b, query_img_p in enumerate(query_img_paths):
                    tmp_out_dir = Path(
                        out_dir,
                        f"r{local_rank}_B{batch_idx:04}_b{b:03}_{query_img_p}",
                        ref_type_short,
                    )
                    tmp_out_dir.mkdir(parents=True, exist_ok=True)
                    tmp_ref_img_paths = [
                        str(Path(*Path(p).parts[-5:])).replace("/", "_").replace(".png", "")
                        for p in ref_img_paths[b]  # (N_ref, )
                    ]

                    # get attn maps of the centre patch in the query image
                    # (B, H, W, N_ref, H, W)
                    attn_weights_map = batch_output[f"attn_weights_map_ref_{ref_type_short}"]
                    tmp_h, tmp_w = attn_weights_map.shape[1:3]

                    if check_patch_mode == "centre":
                        query_patch = (tmp_h // 2, tmp_w // 2)
                    elif check_patch_mode == "random":
                        query_patch = (
                            np.random.randint(0, tmp_h),
                            np.random.randint(0, tmp_w),
                        )
                    else:
                        raise ValueError(f"Unknown check_patch_mode: {check_patch_mode}")
                    attn_weights_map = attn_weights_map[b][query_patch]  # (N_ref, H, W)

                    # write attn maps
                    for ref_idx, (ref_img_p, attn_m) in enumerate(
                        zip(tmp_ref_img_paths, attn_weights_map)
                    ):
                        tmp_out_name = f"ref{ref_idx:02}_{ref_img_p}.png"
                        tmp_out_path = Path(tmp_out_dir, tmp_out_name)
                        attn_m = attn2rgb(attn_m.cpu().numpy())  # (H, W, 3)
                        Image.fromarray(attn_m).save(tmp_out_path)

    def _write_a_score_map_with_colour_mode(self, out_path, score_map):
        if self.write_config.score_map_colour_mode == "gray":
            metric_map_write(out_path, score_map, self.vrange_intrinsic)
        elif self.write_config.score_map_colour_mode == "rgb":
            rgb = gray2rgb(score_map, self.vrange_vis)
            Image.fromarray(rgb).save(out_path)
        else:
            raise ValueError(f"colour_mode {self.write_config.score_map_colour_mode} not supported")
