from pathlib import Path
from yolov5 import run

weights = Path("./pretrain_weights/best.pt")
source = 0
run(weights=weights, source=source)
