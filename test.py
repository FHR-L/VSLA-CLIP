import os
from config import cfg
import argparse
from datasets.make_video_dataloader import make_dataloader
from model.make_model_clipvideoreid_reidadapter_pbp import make_model
from utils.test_video_reid import test
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    (train_loader_stage2, train_loader_stage1,
     query_loader, gallery_loader,
     queryloader_all_frames, galleryloader_all_frames,
     num_classes, num_query, camera_num) = make_dataloader(cfg)

    # TODO
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=0)
    model.load_param(cfg.TEST.WEIGHT)

    use_gpu = True
    cmc, mAP, ranks = test(model, query_loader, gallery_loader, use_gpu, cfg)
    ptr = "mAP: {:.2%}".format(mAP)
    for r in ranks:
        ptr += " | R-{:<3}: {:.2%}".format(r, cmc[r - 1])
    logger.info(ptr)