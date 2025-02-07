"""
Adapted from Grounded SAM2 demo file
"""

import os
import cv2
import hydra
from omegaconf import OmegaConf
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
import scipy.ndimage
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict, load_image
from torchvision.ops import box_convert
import argparse
from typing import Tuple, List, Union

# Example usage: python gsam_wrapper.py --video_path /home/jacinto/robot-grasp/data/demos/simple_movements/18/video.mp4 --output_dir /home/jacinto/robot-grasp/data/demos/simple_movements/18/ --object_name "mug."


"""
Hyper parameters
"""
CURRENT_DIR = Path(__file__).parent
SAM2_CHECKPOINT = CURRENT_DIR / "checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = CURRENT_DIR / "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = CURRENT_DIR / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = CURRENT_DIR / "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# environment settings
# use bfloat16

def build_sam2_direct(config, checkpoint_path, device):
    model = hydra.utils.instantiate(config.model)
    if checkpoint_path is not None:
        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        # Remove 'model' prefix if present
        if "model" in state_dict:
            state_dict = state_dict["model"]
            
        # Load state dict with strict=False to ignore missing keys
        model.load_state_dict(state_dict, strict=False)
        print("Loaded checkpoint with some missing keys - this is expected")
    
    model.to(device)
    return model


class GSAM2:
    def __init__(self,
                 device = None,
                 output_dir = Path("outputs/"),
                 debug = False):
        
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug

        # build SAM2 image predictor
        sam2_config_path = SAM2_MODEL_CONFIG
        sam2_cfg = OmegaConf.load(sam2_config_path)
        self.sam2_model = build_sam2_direct(sam2_cfg, SAM2_CHECKPOINT, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # build grounding dino model
        self.grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=self.device
        )
    
    def load_video_frame(self, video_path, frame = 0):
        
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        # Set the video to the desired frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    
        ret, image = cap.read()
        if not ret:
            raise ValueError(f"Cannot read frame {frame}")
    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_source = Image.fromarray(image)
        image_transformed, _ = transform(image_source, None)

        return image, image_transformed
    
    def get_masks_video(self, object_names, video_path, frame = 0):

        image_source, image = self.load_video_frame(video_path, frame)

        self.sam2_predictor.set_image(image_source)

        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=image,
            caption=object_names,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # FIXME: figure how does this influence the G-DINO model
        torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if len(masks.shape) == 3:
            masks = masks[np.newaxis, :]

        return masks, scores, logits, confidences, labels, input_boxes
    
    def get_masks_image(self, object_names: str, image: Union[str, np.ndarray]):
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        if isinstance(image, str):
            image_source, image = load_image(image_path=image)
        else:
            image_source, image = load_image(image_array=image)

        self.sam2_predictor.set_image(image_source)

        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=image,
            caption=object_names,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # FIXME: figure how does this influence the G-DINO model
        torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        try: 
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            if len(masks.shape) == 3:
                masks = masks[np.newaxis, :]

            if self.debug:
                self.visualize_image(image_source, masks, scores, labels, input_boxes)

            return masks, scores, logits, confidences, labels, input_boxes
        
        except AssertionError as e:
            print(f"GSAM 2 failed to generate masks for object: {object_names}")
            print(f"AssertionError: {e}")
            return None, None, None, None, None, None

    
    def first_n_unique_elements(self, strings, n=3):
        """
        Get the first n unique elements from a list.

        Args:
            strings (list): List of strings.
            n (int): Number of unique elements to extract.

        Returns:
            list: List of the first n unique elements.
        """
        seen = set()
        unique_elements = []
        unique_indices = []

        for i, string in enumerate(strings):
            if string not in seen:
                seen.add(string)
                unique_elements.append(string)
                unique_indices.append(i)
                if len(unique_elements) == n:
                    break

        return unique_elements, unique_indices
    
    def filter_masks(self, masks, labels, num_objects, technique="closest_to_center"):

        if technique == "first_n":
            unique_labels, indices = self.first_n_unique_elements(labels, num_objects)        # We only take the best confidence for each, assuming confidences are already in descending order
            filtered_masks = masks[indices]
        if technique == "closest_to_center":
            print("Filtering masks using closest to center technique")

            def find_centroid(mask):
                return scipy.ndimage.center_of_mass(mask)

            def distance_from_center(centroid, center):
                return np.sqrt((centroid[0] - center[0]) ** 2 + (centroid[1] - center[1]) ** 2)

            h, w = masks.shape[2], masks.shape[3]
            center = (h / 2, w / 2)

            unique_labels, indices = self.first_n_unique_elements(labels, num_objects)
            filtered_masks = []

            for label in unique_labels:
                label_indices = [i for i, l in enumerate(labels) if l == label]
                label_masks = masks[label_indices]
                
                centroids = [find_centroid(mask[0]) for mask in label_masks]
                distances = [distance_from_center(centroid, center) for centroid in centroids]
                
                closest_index = label_indices[np.argmin(distances)]
                filtered_masks.append(masks[closest_index])

            filtered_masks = np.array(filtered_masks)
        
        return filtered_masks, unique_labels
    
    def erode_masks(self, masks, kernel_size = 10):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded_masks = []

        # Handle (N, 1, H, W) shape by squeezing the second dimension
        masks = np.squeeze(masks, axis=1)

        for mask in masks:
            mask = mask.astype(np.uint8)
            eroded_mask = cv2.erode(mask, kernel, iterations=1)
            # Restore the shape to match input format
            eroded_mask = np.expand_dims(eroded_mask, axis=0)
            eroded_masks.append(eroded_mask)
        
        if self.debug:
            for i, (original_mask, eroded_mask) in enumerate(zip(masks, eroded_masks)):
                original_mask_img = original_mask * 255
                eroded_mask_img = np.squeeze(eroded_mask) * 255
                cv2.imwrite(os.path.join(self.output_dir, f"original_mask_{i}.png"), original_mask_img)
                cv2.imwrite(os.path.join(self.output_dir, f"eroded_mask_{i}.png"), eroded_mask_img)
        
        return np.array(eroded_masks)
    
    def save_masks(self, masks, output_path):
        """
        Save masks as binary images
        """
        for i, mask in enumerate(masks):
            mask_img = np.squeeze(mask) * 255
            cv2.imwrite(os.path.join(output_path, f"mask_{i}.png"), mask_img)
    
    def visualize_video_frame_masks(self, video_path, masks, confidences, labels, input_boxes, frame = 0):

        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences = confidences.numpy().tolist()
        class_names = labels

        class_ids = np.array(list(range(len(class_names))))

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]

        """
        Visualize image with supervision useful API
        """
        # img = cv2.imread(img_path)
        # cv2.imwrite("img.jpg", img)

        # np.save("duck_mask.npy", masks[0])

        img, _ = self.load_video_frame(video_path, frame)

        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(self.output_dir, "groundingdino_annotated_image.jpg"), annotated_frame)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(self.output_dir, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)


    def visualize_image(self, image, masks, confidences, labels, input_boxes):

        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Handle different input types for confidences
        if torch.is_tensor(confidences):
            confidences = confidences.cpu().numpy()
        confidences = np.ravel(confidences).tolist()

        class_names = labels

        class_ids = np.array(list(range(len(class_names))))

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]

        """
        Visualize image with supervision useful API
        """
        # img = cv2.imread(img_path)
        # cv2.imwrite("img.jpg", img)

        # np.save("duck_mask.npy", masks[0])

        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(self.output_dir, "groundingdino_annotated_image.jpg"), annotated_frame)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(self.output_dir, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

        # Display the annotated frames
        cv2.imshow('GroundingDINO Annotated Image', annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GSAM2 Mask Generator')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='/home/jacinto/robot-grasp/data/demos/spatial_tracker_testing/',
                        help='Output directory for results')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--frame', type=int, default=0, help='Frame number to process')
    parser.add_argument('--object_name', type=str, required=True, help='Object name to detect')

    args = parser.parse_args()

    gsam2 = GSAM2(
        device=args.device,
        output_dir=Path(args.output_dir),
        debug=args.debug
    )
    
    # Get masks and required data for visualization
    masks, scores, logits, confidences, labels, input_boxes = gsam2.get_masks(args.object_name, args.video_path, frame=args.frame)
    
    # Filter the masks
    filtered_masks, filtered_labels = gsam2.filter_masks(masks, labels, num_objects=3)
    
    # Save the filtered masks
    if args.debug:
        print("Saving masks...")
        gsam2.save_masks(filtered_masks, args.output_dir)