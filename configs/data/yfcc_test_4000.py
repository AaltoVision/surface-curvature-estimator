from configs.data.base import cfg

TEST_BASE_PATH = "assets/yfcc_test_4000"

cfg.DATASET.TEST_DATA_SOURCE = "YFCC"
cfg.DATASET.TEST_DATA_ROOT = "data/yfcc/test"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/yfcc_test_4000.txt"

cfg.DATASET.MGDPT_IMG_RESIZE = 832
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
