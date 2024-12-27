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

    def extract_detections(self, results, img_height, img_width, upper_percentage=0.25):
        detections = []
        upper_threshold = img_height * upper_percentage

        # results is typically a list of arrays or a tuple
        # for each label, we have a list of bboxes
        # each bounding box is [x1, y1, x2, y2, score]
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
        # For more thorough testing, we could replicate or shuffle, but here we just keep it simple
        self.files = 1 * self.files
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


def process_video(args, video_path):
    """
    1. Extract frames to data/{video_name}/images (unless already present)
    2. Run inference on each frame
    3. Overlay bounding boxes and write to data/{video_name}/overlay
    4. Save bounding boxes to a COCO JSON in data/{video_name}/coco_annotations.json
    """
    # --- Prepare paths ---
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    images_dir = f"data/{video_name}/images"
    overlay_dir = f"data/{video_name}/overlay"
    coco_json_path = f"data/{video_name}/coco_annotations.json"

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # --- Check if images exist; if not, extract frames ---
    # A simple check: if images_dir is empty, extract frames
    if not os.listdir(images_dir):
        print(f"No frames found in {images_dir}. Extracting frames from video...")
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(images_dir, f"{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_idx += 1
        cap.release()
        print(f"Extracted {frame_idx} frames to {images_dir}")
    else:
        print(f"Frames already exist in {images_dir}. Skipping extraction.")

    # --- Initialize model once ---
    model = init_detector(args.config, args.checkpoint, device=args.device)
    print("Model loaded for video inference.")

    # --- COCO JSON structure ---
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "player"}
        ]
    }

    annotation_id = 1
    image_id = 1

    # --- Inference on each frame ---
    frame_files = sorted(os.listdir(images_dir))
    for frame_file in frame_files:
        if not frame_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        frame_path = os.path.join(images_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        height, width = frame.shape[:2]

        # Inference
        results = inference_detector(model, frame)

        # Extract detections
        detections = []
        # results[0] when it's a single class or multi-class model output
        for label, bboxes in enumerate(results[0] if isinstance(results, tuple) else results):
            for bbox in bboxes:
                x1, y1, x2, y2, score = bbox[:5]
                if score < 0.05:  # a low threshold to filter out very low scores
                    continue
                w, h = x2 - x1, y2 - y1

                # Save detection
                detections.append({
                    "bbox": [int(x1), int(y1), int(w), int(h)],
                    "score": float(score),
                    "category_id": label + 1,
                })

        # Simple NMS
        detections = nms(detections, iou_threshold=0.75)

        # --- Add to COCO JSON ---
        coco_output["images"].append({
            "file_name": frame_file,
            "height": height,
            "width": width,
            "id": image_id
        })

        for det in detections:
            # bounding box in COCO is [x, y, width, height]
            bbox = det["bbox"]
            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": det["category_id"],
                "bbox": bbox,
                "score": det["score"],
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            annotation_id += 1

        # --- Overlay bounding boxes on frame ---
        for det in detections:
            x, y, w, h = det["bbox"]
            score = det["score"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save overlaid frame
        overlay_path = os.path.join(overlay_dir, frame_file)
        cv2.imwrite(overlay_path, frame)

        image_id += 1

    # --- Write out COCO JSON ---
    with open(coco_json_path, "w") as f:
        json.dump(coco_output, f, indent=2)

    print(f"Finished inference on video. Bboxes saved to {coco_json_path}")
    print(f"Overlay frames saved to {overlay_dir}")


def nms(detections, iou_threshold=0.75):
    """
    A standalone NMS function for bounding box dictionary list:
    detections = [{"bbox": [x, y, w, h], "score": float, "category_id": int}, ...]
    """
    if not detections:
        return []

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
            iou = compute_iou_np(boxes[i], boxes[j])
            ious.append(iou)
        ious = np.array(ious)
        indices = indices[1:][ious < iou_threshold]

    return [detections[i] for i in keep]


def compute_iou_np(box1, box2):
    """
    box1, box2: [x, y, w, h]
    """
    x1_1, y1_1, w1, h1 = box1
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1

    x1_2, y1_2, w2, h2 = box2
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    iou = inter_area / float(area1 + area2 - inter_area + 1e-6)
    return iou


def main():
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--safety-model', help='Safety model to be used for inference', default='hockey')
    parser.add_argument('--video-path', help='Path to the input video', default='assets/IMG-2113.mov') # use mp4's
    args = parser.parse_args()

    # Set the start method
    set_start_method('spawn', force=True)

    # Specify the config file and checkpoint file
    args.device = 'cuda'
    args.config = 'configs/co_dino_hockey/co_dino_5scale_swin_large_16e_o365tococo.py'
    args.checkpoint = 'pretrained/co_dino_5scale_swin_large_16e_o365tococo.pth' #'latest.pth'

    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.info(f'Inference worker for {args.safety_model} initialized')

    # If we have a video path, run single-video inference
    if args.video_path:
        print(f"Processing single video: {args.video_path}")
        process_video(args, args.video_path)
        return

    # Otherwise, run the multi-process test channel approach
    # This is only for the case that the image is not found in the inference_images folder (EFS)
    os.makedirs("inference_images/images", exist_ok=True)

    queue_name = "safety-worker-hockey"
    timeout = 60  # 60 seconds in this example
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
