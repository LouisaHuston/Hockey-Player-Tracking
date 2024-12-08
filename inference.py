
from mmdet.apis import init_detector, inference_detector

from multiprocessing import Process, set_start_method, Value, Lock
from argparse import ArgumentParser
from datetime import timezone

import numpy as np

import colorlog
import datetime
import logging
import torch
import time
import json
import cv2
import os

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--safety-model', help='Safety model to be used for inference', default='hockey', required=False)
    args = parser.parse_args()
    return args

def get_utc_timestamp():
    dt = datetime.datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    utc_timestamp = utc_time.timestamp()
    return utc_timestamp

class InferenceWorker:
    def __init__(self, args, channel, queue_name, timeout):
        self.args = args
        self.channel = channel
        self.queue_name = queue_name
        self.timeout = timeout
        self.class_names = {'id': 'player'}
        self.id_db_map = {1: 1}
        self.logger = self.setup_logger()
        self.logger.info('InferenceWorker initialized')
        self.total_images = 0
        self.total_time = 0
        self.seen_images = set()
        self.start_time = time.time()
        self.load_model()

    def setup_logger(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s: %(name)s: %(message)s",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        ))
        logger = colorlog.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        return logger

    def load_model(self):
        self.model = init_detector(self.args.config, self.args.checkpoint, device=self.args.device)
        self.logger.info('Model loaded')

    def infer(self, image_path):
        if image_path is None:
            raise ValueError("image_path is None")

        # Check if the file exists
        if not os.path.exists(image_path):
            raise ValueError(f"Image file does not exist: {image_path}")

        return inference_detector(self.model, cv2.imread(image_path))

    def nms(self, detections, iou_threshold=0.75):
        if not detections:
            return []

        # Convert bounding boxes to numpy arrays
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['score'] for d in detections])
        indices = np.argsort(scores)[::-1]

        keep = []
        while indices.size > 0:
            i = indices[0]
            keep.append(i)
            if indices.size == 1:
                break

            ious = []
            for j in indices[1:]:
                iou = self.compute_iou(boxes[i], boxes[j])
                ious.append(iou)
            ious = np.array(ious)
            indices = indices[1:][ious < iou_threshold]

        return [detections[i] for i in keep]

    def compute_iou(self, box1, box2):
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = w1 * h1
        box2_area = w2 * h2

        iou = inter_area / (box1_area + box2_area - inter_area)
        return iou

    def calculate_bbox_area(self, x1, y1, x2, y2, image_height, image_width):
        # Calculate the width and height of the bounding box
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        scaled_bbox_area = bbox_area / (image_height * image_width)
        return scaled_bbox_area

    def extract_detections(self, results, img_height, image_width, upper_percentage=0.25):
        detections = []
        upper_threshold = img_height * upper_percentage

        for label, bboxes in enumerate(results[0] if isinstance(results, tuple) else results):
            for bbox in bboxes:
                x1, y1, x2, y2, score = bbox[:5]
                width, height = x2 - x1, y2 - y1
                cat_id = int(label) + 1
                
                detections.append({
                    "bbox": [int(x1), int(y1), int(width), int(height)],
                    "score": float(score),
                    "category_id": cat_id,
                    "category_name": 'player',
                    "db_id": 1,
                })

        # Perform NMS
        filtered_detections = []
        categories = set(d['category_id'] for d in detections)
        for category in categories:
            category_detections = [d for d in detections if d['category_id'] == category]
            filtered_detections.extend(self.nms(category_detections))

        return filtered_detections

    def print_summary(self):
        if self.total_images > 0:
            average_time = self.total_time / self.total_images
            self.logger.info(f"{self.args.device} | Total images processed: {self.total_images}")
            self.logger.info(f"{self.args.device} | Total time taken: {self.total_time:.2f} seconds")
            self.logger.info(f"{self.args.device} | Average time per image: {average_time:.2f} seconds")
        else:
            self.logger.info(f"{self.args.device} | No images were processed.")

    def run_test(self):
        self.channel.basic_qos(prefetch_count=1)

        try:
            while True:
                method_frame, header_frame, body = self.channel.basic_get(self.queue_name)
                if method_frame:
                    data = json.loads(body)
                    self.channel.basic_ack(data['id'])
                    image_path = f"data/images/{data['base_image_key']}"
                    start_inference_time = time.time()
                    results = self.infer(image_path)
                    img_height, img_width = cv2.imread(image_path).shape[:2]
                    inference_time = time.time() - start_inference_time
                    self.total_images += 1
                    self.total_time += inference_time
                    detections = self.extract_detections(results, img_height, img_width)

                    for detection in detections:
                        self.logger.info(f"GPU {self.args.device} | Finished Image {data['id']} | label: {detection['category_name']} | score: {detection['score']}")

                    self.logger.info(f"GPU {self.args.device} | Image {data['base_image_key']} processed")
                    self.start_time = time.time()

                else:
                    if time.time() - self.start_time > self.timeout:
                        self.logger.info("No messages in the queue for 60 seconds. Exiting")
                        break

        except Exception as e:
            self.logger.error(f"Error in running the worker: {e}")
            raise Exception(f"Error in running the worker: {e}")

        self.print_summary()

def worker_process_test(args, gpu_id, timeout, queue_name, logger):
    args.device = f'cuda:{gpu_id}'

    channel = TestChannel(f'data/images')
    worker = InferenceWorker(args, channel, queue_name, timeout)
    worker.run_test()

class TestChannel:
    def __init__(self, folder):
        self.files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        self.files = 1 * self.files # For increased testing
        self.index = Value('i', 0)
        self.acknowledged = set()
        self.lock = Lock()

    def basic_qos(self, prefetch_count):
        pass

    def basic_get(self, queue_name):
        with self.lock:  # Ensure that index is updated atomically
            while self.index.value < len(self.files):
                delivery_tag = self.index.value + 1
                if delivery_tag not in self.acknowledged:
                    filename = self.files[self.index.value]
                    self.index.value += 1
                    message = json.dumps({'id': delivery_tag, 'base_image_key': filename})
                    return (delivery_tag, None, message)
            return (None, None, None)

    def basic_ack(self, delivery_tag):
        with self.lock:
            self.acknowledged.add(delivery_tag)

def main():

    # Parse arguments
    args = parse_args()

    # Set the start method
    set_start_method('spawn')

    # Specify the config file and checkpoint file
    args.device = 'cuda'
    args.config = f'configs/co_dino_hockey/co_dino_5scale_swin_large_16e_o365tococo.py'
    args.checkpoint = f'latest.pth'

    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.info(f'Inference worker for {args.safety_model} initialized')

    # This is only for the case that the image is not found in the inference_images folder (EFS)
    os.makedirs(f"inference_images/images", exist_ok=True)

    # Specify the queue name and timeout
    queue_name = f"safety-worker-hockey"
    timeout = 60 #60*60*4
    num_gpus = torch.cuda.device_count()

    processes = []
    for gpu_id in range(num_gpus):
        p = Process(target=worker_process_test, args=(args, gpu_id, timeout, queue_name, logger))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
