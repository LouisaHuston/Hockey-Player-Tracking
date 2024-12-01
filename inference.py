#!/usr/bin/env python
#-*- coding: utf-8 -*-
#
# ==============================================================================
# Copyright (c) 2024 Nextera Robotic Systems
# Description: Inference script for running inference on images using MMdetection
#
# Author: Andrew Kent
# Created on: Thu Mar 07 2024 5:47:50 PM
# ==============================================================================

from mmdet.apis import init_detector, inference_detector

from multiprocessing import Process, set_start_method, Value, Lock
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser
from dotenv import load_dotenv
from datetime import timezone

import numpy as np

import colorlog
import requests
import datetime
import logging
import redis
import boto3
import torch
import time
import pika
import json
import cv2
import os

load_dotenv()

BASE_ENDPOINT = os.environ.get("NEXTERA_API_HOST")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--safety-model', help='Safety model to be used for inference', default='gh', required=False)
    args = parser.parse_args()
    return args

def download_image_from_s3(bucket, base_image_key, folder_path="temp_inference_images/images"):
    """
    Download the image from the s3 bucket, preserving the full path structure.
    """
    # Construct the full directory path
    full_path = os.path.join(folder_path, base_image_key)
    directory = os.path.dirname(full_path)

    # Create the directory structure if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Download the file from S3
    s3 = boto3.client('s3')
    s3.download_file(bucket, base_image_key, full_path)

    return full_path

def upload_label(box, score, image_json, label_type_id):
    """
    Upload inference results to the database
    using the api
    """
    headers = {"x-api-key": os.environ.get("NEXTERA_API_KEY")}
    string_list = [str(num) for num in box]
    results = {
        'processed_image': image_json["id"],
        'coordinates': f"{ ', '.join(string_list)}",
        'confidence': float(score),
        'shape': 'RECTANGLE',
        'type': label_type_id
    }
    res = requests.post(f"{os.environ.get('NEXTERA_API_HOST')}/labels",
                        json=results, headers=headers)
    if not res.ok:
        raise Exception(f"Failed uploading label to API {image_json['id']}")
    return res.status_code

def mark_model_processed(body, image):
    headers = {"x-api-key": os.environ.get("NEXTERA_API_KEY")}
    resp = requests.patch(f"{BASE_ENDPOINT}/processed-image/{image['id']}", data=body, headers=headers)

    if resp.status_code != 200:
        raise Exception(f"[Mark Processed Failed] {BASE_ENDPOINT}/processed-image/{image['id']}")
    return resp.status_code

def get_utc_timestamp():
    dt = datetime.datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    utc_timestamp = utc_time.timestamp()
    return utc_timestamp

def update_is_processing(key):
    db = redis.Redis(host=os.environ.get("REDIS_DB_URL"), port=6379, password=os.environ.get("REDIS_DB_PASSWORD") ,decode_responses=True)
    db.hset(key, 'is_processing', 1)
    db.hset(key, 'processing_on', str(get_utc_timestamp()))

def update_processed(key):
    db = redis.Redis(host=os.environ.get("REDIS_DB_URL"), port=6379, password=os.environ.get("REDIS_DB_PASSWORD") ,decode_responses=True)
    db.hset(key, 'is_processed', 1)
    db.hset(key, 'processed_on', str(get_utc_timestamp()))

def get_consumer(url, logger):

    parameters = pika.URLParameters(url)
    try:
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        return channel
    except Exception as e:
        logger.error(f"Error in connecting to the queue: {e}")
        raise Exception(f"Error in connecting to the queue: {e}")

def get_instance_id():
    response = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
    instance_id = response.text
    return instance_id

def name_instance(safety_model):
    ec2 = boto3.client('ec2')
    response = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
    instance_id = response.text
    return ec2.create_tags(
        Resources=[
            instance_id,
        ],
        Tags=[
            {
                'Key': 'Name',
                'Value': f"Safety-{safety_model.upper()}-Worker-{instance_id}",
            },
        ],
    )


def notify(safety_model, status="Failed"):
    current = datetime.datetime.now()
    message = {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"SAFETY | {safety_model.upper()} Worker: {get_instance_id()}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Started at:*\n{current.strftime('%Y-%m-%d %H:%M:%S')} UTC"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Status:*\n{status}"
                    }
                ]
            }
        ]
    }
    # definitely need to update the slack webhook NOT SECURE
    resp = requests.post(os.environ.get("SLACK_WEBHOOK_URL"), json=message)
    return resp

class InferenceWorker:
    def __init__(self, args, channel, queue_name, timeout):
        self.args = args
        self.channel = channel
        self.queue_name = queue_name
        self.timeout = timeout
        self.class_names = {c['id']: c['name'] for c in json.load(open(f"configs/co_dino_{self.args.safety_model}/db_id.json", 'r'))}
        self.id_db_map = {c['id']: c['db_id'] for c in json.load(open(f"configs/co_dino_{self.args.safety_model}/db_id.json", 'r'))}
        self.contradictions = json.load(open(f"configs/contradictions/contradictions.json", 'r'))
        self.lookup_table = json.load(open(f"configs/co_dino_{self.args.safety_model}/lookup_table.json", 'r'))
        self.name_threshold_map = {c['id']: c['threshold'] for c in json.load(open(f"configs/co_dino_{self.args.safety_model}/db_id.json", 'r'))}
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

    # Get precision value from lookup table
    def get_precision(self, lookup_table, category, confidence):
        index = min(int(confidence // 0.004016064257028112), 249)
        return lookup_table[str(category)]['precision'][index]

    def are_contradicting(self, cat1, cat2, contradictions, category_map):
        name1 = category_map.get(cat1, "")
        name2 = category_map.get(cat2, "")
        return (name1, name2) in contradictions or (name2, name1) in contradictions

    def remove_overlapping_contradictions(self, detections, iou_threshold=0.75):
        # Return an empty list if there are no detections
        if not detections:
            return []

        # Convert bounding boxes, scores, and categories to numpy arrays
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['score'] for d in detections])
        categories = np.array([d['category_id'] for d in detections])

        # Sort indices based on scores in descending order
        indices = np.argsort(scores)[::-1]

        keep = []  # List to keep track of indices to retain
        discarded = set()  # Set to track discarded indices

        for i in indices:
            if i in discarded:
                continue  # Skip if the index is already discarded

            keep.append(i)  # Add the current index to the keep list

            for j in indices:
                if i == j or j in discarded:
                    continue  # Skip comparing the box with itself or if already discarded

                # Compute IoU between the current box and the others
                iou = self.compute_iou(
                    [boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]],
                    [boxes[j][0], boxes[j][1], boxes[j][0] + boxes[j][2], boxes[j][1] + boxes[j][3]]
                )

                if iou > iou_threshold:
                    # Check if categories are contradicting
                    if self.are_contradicting(categories[i], categories[j], self.contradictions, self.class_names):
                        # Get precision values for the categories
                        precision1 = self.get_precision(self.lookup_table, categories[i], scores[i])
                        precision2 = self.get_precision(self.lookup_table, categories[j], scores[j])
                        if precision1 < precision2:
                            discarded.add(i)  # Discard the current box if it has lower precision
                            keep.remove(i)
                            break
                        else:
                            discarded.add(j)  # Discard the overlapping box if it has lower precision

        # Return the detections corresponding to the kept indices
        return [detections[i] for i in keep]



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

    def calculate_phi(self, y1, y2, image_height):
        # Calculate the height of the bounding box
        bbox_height = y2 - y1

        # Calculate the midpoint of the bounding box
        y_mid = y1 + (bbox_height / 2)

        # Calculate phi based on the midpoint
        phi = -((y_mid - image_height) / image_height) * np.pi

        return phi

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

                # Filter out detections that are in the upper n% of the image
                if y1 < upper_threshold:
                    continue

                if score < self.name_threshold_map[cat_id]:
                    continue
                # hands and gloves: phi filters out bottom angle and area excludes large bbox
                if self.args.safety_model == "ppe" and cat_id in [1,2]:
                    phi = self.calculate_phi(y1, y2, img_height)
                    bbox_area = self.calculate_bbox_area( x1, y1, x2, y2, img_height, image_width)
                    if phi < 0.51373182 or bbox_area > .003:
                        continue


                detections.append({
                    "bbox": [int(x1), int(y1), int(width), int(height)],
                    "score": float(score),
                    "category_id": cat_id,
                    "category_name": self.class_names[cat_id],
                    "db_id": self.id_db_map[cat_id],
                })

        detections = self.remove_overlapping_contradictions(detections)
        # Perform NMS
        filtered_detections = []
        categories = set(d['category_id'] for d in detections)
        for category in categories:
            category_detections = [d for d in detections if d['category_id'] == category]
            # Skips Guardrail category when performing non-maximal suppression
            if self.args.safety_model == "gh" and category in [1,2,3,4,5,6,7,8,12]:
                filtered_detections.extend(category_detections)
            else:
                filtered_detections.extend(self.nms(category_detections))

        return filtered_detections

    def run(self):
        self.channel.basic_qos(prefetch_count=1)
        self.start_time = time.time()

        try:
            while True:
                with ThreadPoolExecutor() as executor:
                    method_frame, header_frame, body = self.channel.basic_get(self.queue_name)
                    if method_frame:
                        self.channel.basic_ack(method_frame.delivery_tag)
                        data = json.loads(body)
                        image_name = data['base_image_key']

                        if image_name in self.seen_images:
                            self.logger.info(f"Image {image_name} has been processed before.")
                        else:
                            self.seen_images.add(image_name)

                            update_is_processing(f"{data['id']}:{self.args.safety_model}")
                            image_path = f"data/{image_name}"
                            delete_image = False

                            if not os.path.exists(image_path):
                                self.logger.error(f"Image {data['id']} not found")
                                default_bucket = os.environ.get('DEFAULT_S3_BUCKET')
                                image_path = download_image_from_s3(default_bucket, image_name)
                                delete_image = True

                            start_inference_time = time.time()
                            results = self.infer(image_path)
                            inference_time = time.time() - start_inference_time
                            self.total_images += 1
                            self.total_time += inference_time
                            self.logger.info(f"Image {image_name} has {len(results[0])} labels")

                            # Run the rest asynchronously
                            executor.submit(self.process_detections, results, data, image_name, delete_image, image_path)
                    else:
                        if time.time() - self.start_time > self.timeout:
                            self.logger.info("No messages in the queue for 60 seconds. Exiting")
                            break

        except Exception as e:
            self.logger.error(f"Error in running the worker: {e}")
            raise Exception(f"Error in running the worker: {e}")

        self.print_summary()

    def process_detections(self, results, data, image_name, delete_image, image_path):
        img_height, img_width = cv2.imread(image_path).shape[:2]
        detections = self.extract_detections(results, img_height, img_width)
        self.logger.info(f"{self.args.device} | Uploaded Image {data['id']}")

        for detection in detections:
            upload_label(detection['bbox'], detection['score'], data, detection['db_id'])
            self.logger.info(f"{self.args.device} | Uploaded Image {data['id']} | label: {detection['category_name']} | score: {detection['score']}")

        update_processed(f"{data['id']}:{self.args.safety_model}")
        mark_model_processed({f"{self.args.safety_model}_processed": True}, data)
        self.logger.info(f"Image {image_name} processed")

        if delete_image:
            os.remove(image_path)

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

def worker_process(args, gpu_id, timeout, queue_name, logger):
    args.device = f'cuda:{gpu_id}'
    channel = get_consumer(os.environ.get('RABIT_MQ_URL'), logger)
    worker = InferenceWorker(args, channel, queue_name, timeout)
    worker.run()

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

    # Testing flag
    testing = False

    # Set the start method
    set_start_method('spawn')

    # Name the instance
    if not testing:
        name_instance(args.safety_model)

    # Specify the config file and checkpoint file
    args.device = 'cuda'
    args.config = f'configs/co_dino_{args.safety_model}/co_dino_5scale_swin_large_16e_o365tococo.py'
    args.checkpoint = f'{args.safety_model}/best_bbox_mAP.pth'

    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.info(f'Inference worker for {args.safety_model} initialized')

    # This is only for the case that the image is not found in the inference_images folder (EFS)
    os.makedirs(f"temp_inference_images/images", exist_ok=True)

    # Specify the queue name and timeout
    queue_name = f"safety-worker-{args.safety_model}"
    timeout = 60 #60*60*4
    num_gpus = torch.cuda.device_count()

    processes = []
    for gpu_id in range(num_gpus):
        if not testing:
            p = Process(target=worker_process, args=(args, gpu_id, timeout, queue_name, logger))
        else:
            p = Process(target=worker_process_test, args=(args, gpu_id, timeout, queue_name, logger))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
