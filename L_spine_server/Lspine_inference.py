import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog

## parameter 설정
dataname = "L-spine"
json_dir = f"./columns"
img_dir = f"./test_imgs"
test_dir = f'./test_imgs'                   # test image 들어간 directory
model_dir = f'./model'                      # L-spine.pth 들어간 directory
output_dir = f'./out'                       # output파일들 저장될 directory


for d in ["train", "validation"]:
    register_coco_instances(f"{dataname}_{d}", {}, f"{json_dir}/{d}.json", img_dir)


MetadataCatalog.get(f"{dataname}_train").set(thing_classes=['VERTEBRA', 'NARROW_DISC_SPACE', 'NORMAL_DISC_SPACE'])

Lspine_metadata = MetadataCatalog.get(f"{dataname}_train")


cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.OUTPUT_DIR = output_dir

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.MODEL.WEIGHTS = os.path.join(model_dir, f"L-spine.pth")  # 학습된 모델 들어가 있는 곳
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # custom testing threshold
predictor = DefaultPredictor(cfg)

test_list = os.listdir(test_dir)
test_list.sort()
except_list = []

for file in tqdm(test_list):
    filepath = os.path.join(test_dir, file)
    filename = os.path.splitext(file)[0]
    im = cv2.imread(filepath)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=Lspine_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = Image.fromarray(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    img.save(f'{output_dir}/pred_{filename}.jpg', dpi=(1000, 1000))

    # plt.figure(figsize = (14, 10))
    # plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    # plt.savefig(f'{output_dir}/pred_{filename}.jpg', dpi=1000)
