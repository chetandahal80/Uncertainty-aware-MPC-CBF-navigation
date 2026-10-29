import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2
from ultralytics import YOLO
import cvzone
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from message_filters import ApproximateTimeSynchronizer, Subscriber
from .state_estimation import *
import time
import torch


class RGBDSubscriber(Node):
    def __init__(self):
        super().__init__('rgbd_subscriber')

        self.subscription_rgb = Subscriber(
            self,
            Image,
            '/camera/image_raw'
        )

        self.subscription_depth = Subscriber(
            self,
            Image,
            '/camera/depth/image_raw'
        )

        self.ts = ApproximateTimeSynchronizer(
            [self.subscription_rgb, self.subscription_depth],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        self.bridge = CvBridge()

        self.model = YOLO(
            "src/human_tracking/human_tracking/"
            "Yolo-weights/yolov8m.pt"
        )

        self.classNames = list(self.model.names.values())
        self.latest_depth_image = None

        self.tracker = Tracker(
            min_iou=0.25,
            min_streak=3,
            max_age=40,
            w_iou=0.8,
            w_depth=0.2
        )

    def get_depth(self, depth_image, bbox_arngd):
        """
        Estimate object depths using the original histogram,
        visibility calculation and iterative masking algorithm.
        """

        depth_image = np.round(depth_image * (255 / 10.0))
        depth_image[depth_image > 255] = 255
        depth_image = depth_image.astype(np.uint8)

        box_count = len(bbox_arngd)
        Th = 0.1
        actual_depth = {}
        delta = 7.7

        mask = np.ones(
            (depth_image.shape[0], depth_image.shape[1])
        )

        while bbox_arngd:
            j = 1
            Cj = {}

            for key in bbox_arngd:
                x1 = bbox_arngd[key][0]
                y1 = bbox_arngd[key][1]
                x2 = bbox_arngd[key][2]
                y2 = bbox_arngd[key][3]

                depth_values = depth_image[y1:y2, x1:x2]
                roi_mask = mask[y1:y2, x1:x2]

                filter_depth = np.where(
                    roi_mask == 1,
                    depth_values,
                    np.nan
                )

                valid_depth_values = filter_depth[
                    ~np.isnan(filter_depth)
                ]

                filtered_depth = valid_depth_values[
                    valid_depth_values <= 254
                ]

                if filtered_depth.size == 0:
                    actual_depth[key] = 10.0
                    continue

                y, x = np.histogram(
                    filtered_depth,
                    bins=75
                )

                min_count = np.min(y)
                max_count = np.max(y)

                if max_count == min_count:
                    actual_depth[key] = 10.0
                    continue

                y = (
                    (y - min_count)
                    / (max_count - min_count)
                )

                # Pad histogram boundaries to detect endpoint peaks.
                padded_y = np.pad(
                    y,
                    pad_width=1,
                    mode="constant",
                    constant_values=0.0
                )

                peaks, _ = find_peaks(
                    padded_y,
                    height=0.025,
                    distance=16
                )

                # Map padded positions back to original bin indices.
                peaks = peaks - 1

                if len(peaks) == 0:
                    actual_depth[key] = 10.0
                    continue

                peak_values = y[peaks]

                sorted_order = np.argsort(
                    peak_values
                )[::-1]

                sorted_indices = peaks[sorted_order]

                peak_values = np.sort(
                    peak_values
                )[::-1]

                F_pd = peak_values[0]

                if len(peak_values) > 1:
                    F_sd = peak_values[1]
                else:
                    F_sd = 0

                pd = x[sorted_indices[0]]

                V = F_pd - F_sd

                if V >= Th:
                    actual_depth[key] = pd * (10.0 / 255)

                    condition = (
                        (depth_image >= (pd - delta))
                        & (depth_image <= (pd + delta))
                    )

                    mask[y1:y2, x1:x2][
                        condition[y1:y2, x1:x2]
                    ] = 0

                else:
                    Cj[key] = bbox_arngd[key]
                    j = j + 1

            M = len(Cj)

            if box_count == M:
                if Th > 0.01:
                    Th = Th / 2
                else:
                    Th = 0

                bbox_arngd = Cj
                box_count = M

            if box_count > M:
                bbox_arngd = Cj
                box_count = M

        print(actual_depth)
        return actual_depth

    def sync_callback(self, rgb_msg, depth_msg):
        try:
            start = time.time()

            img = self.bridge.imgmsg_to_cv2(
                rgb_msg,
                desired_encoding='bgr8'
            )

            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding='32FC1'
            )

            self.latest_depth_image = np.array(
                depth_image,
                dtype=np.float32
            )

            img_copy = self.latest_depth_image.copy()

            results = self.model(
                img,
                stream=True,
                device=0,
                quantize=16,
                imgsz=480,
                verbose=False
            )

            for r in results:
                box_depth = {}
                class_names = []
                bbox = {}
                state = []
                boxes = r.boxes

                if len(boxes) == 0:
                    print("No object detected")
                    break

                i = 1

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]

                    x1, y1, x2, y2 = (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    )

                    w = x2 - x1
                    h = y2 - y1

                    conf = (
                        math.ceil(box.conf[0] * 100)
                        / 100
                    )

                    clss = int(box.cls[0])

                    if (
                        conf > 0.4
                        and self.classNames[clss] == 'person'
                    ):
                        cvzone.cornerRect(
                            img,
                            [x1, y1, w, h]
                        )

                        if self.latest_depth_image is not None:
                            depth_values = (
                                self.latest_depth_image[
                                    y1:y2,
                                    x1:x2
                                ]
                            )

                            norm_depth_values = np.round(
                                depth_values * (255 / 10.0)
                            )

                            flattened_array = (
                                norm_depth_values.flatten()
                            )

                            filtered_array = flattened_array[
                                flattened_array <= 255
                            ]

                            box_depth[f'b{i}'] = filtered_array

                            bbox[f'b{i}'] = [
                                x1,
                                y1,
                                x2,
                                y2,
                                conf
                            ]

                            class_names.append(
                                self.classNames[clss]
                            )

                            i = i + 1

                sorted_depth = {
                    k: v
                    for k, v in sorted(
                        box_depth.items(),
                        key=lambda item: (
                            np.mean(item[1])
                            if item[1].size > 0
                            else float('inf')
                        )
                    )
                }

                bbox_ordered = {
                    key: bbox[key]
                    for key in sorted_depth
                }

                startt = time.time()

                depth = self.get_depth(
                    img_copy,
                    bbox_ordered
                )

                endtt = time.time()

                for key in bbox:
                    combined = bbox[key] + [depth[key]]
                    state.append(combined)

                state = np.array(state)

                if state.size == 0:
                    state = np.empty((0, 6))
                else:
                    state[:, [4, 5]] = state[:, [5, 4]]

                results_tracker = self.tracker.tracking(
                    state
                )

                print(
                    f"Took {(endtt - startt):.6f} "
                    "for tracking"
                )

                for result in results_tracker:
                    xx1, yy1, xx2, yy2, dd, Id = result

                    xx1, yy1, xx2, yy2, dd = (
                        int(xx1),
                        int(yy1),
                        int(xx2),
                        int(yy2),
                        round(dd, 2)
                    )

                    ww = xx2 - xx1
                    hh = yy2 - yy1

                    cvzone.cornerRect(
                        img,
                        [xx1, yy1, ww, hh],
                        l=9,
                        rt=2,
                        colorR=(255, 0, 0)
                    )

                    cvzone.putTextRect(
                        img,
                        f'{Id} {dd}',
                        (
                            max(0, xx1),
                            max(25, yy1)
                        ),
                        scale=1,
                        thickness=1,
                        offset=5
                    )

                cv2.imshow("Image", img)
                cv2.waitKey(1)

            end = time.time()

            print(
                f"Took {(end - start):.6f} "
                "for one frame"
            )

        except CvBridgeError as e:
            self.get_logger().error(
                f'CvBridge Error: {str(e)}'
            )

        except Exception as e:
            self.get_logger().error(
                f'An error occurred: {e}'
            )


def main(args=None):
    rclpy.init(args=args)

    rgbd_subscriber = RGBDSubscriber()
    rclpy.spin(rgbd_subscriber)

    rgbd_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
