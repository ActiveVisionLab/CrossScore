def check_metric_prediction_config(
    metric_type,
    metric_min,
    metric_max,
):
    valid_max = False
    valid_min = False
    valid_type = False

    if metric_type in ["ssim", "mse", "mae"]:
        valid_type = True

    if metric_max == 1:
        valid_max = True

    if metric_type == "ssim":
        if metric_min in [-1, 0]:
            valid_min = True
    elif metric_type in ["mse", "mae"]:
        if metric_min == 0:
            valid_min = True

    if not valid_type:
        raise ValueError(f"Invalid metric type {metric_type}")

    valid_range = valid_min and valid_max
    if not valid_range:
        raise ValueError(f"Invalid metric range {metric_min} to {metric_max} for {metric_type}")


def check_reference_type(do_reference_cross):
    if do_reference_cross:
        ref_type = "cross"
    else:
        raise ValueError("Reference type must be 'cross'")
    return ref_type


class ConfigChecker:
    """
    Check if a config object is valid for
        - train/val/test/predict steps that correspond to the lightning module;
        - dataloader creation.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def _check_common_lightning(self):
        check_reference_type(self.cfg.model.do_reference_cross)
        check_metric_prediction_config(
            self.cfg.model.predict.metric.type,
            self.cfg.model.predict.metric.min,
            self.cfg.model.predict.metric.max,
        )

    def check_train_val(self):
        self._check_common_lightning()

    def check_test(self):
        self._check_common_lightning()

    def check_predict(self):
        self._check_common_lightning()

    def check_dataset(self):
        check_metric_prediction_config(
            self.cfg.model.predict.metric.type,
            self.cfg.model.predict.metric.min,
            self.cfg.model.predict.metric.max,
        )
