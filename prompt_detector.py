import torch
import cv2
import numpy as np
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class PromptedDetector:
    def __init__(self, model_name="google/owlvit-base-patch32", device=None):
        print("[INFO] Initializing Prompt-based Detection (OwlViT)...")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and processor
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print(f"[INFO] OwlViT initialized successfully — ready for zero-shot detection on {self.device}!")

    @torch.inference_mode()
    def detect(self, image_bgr, prompt=["a deer", "an animal", "a person", "a car", "a truck", "a motorbike", "a dog", "a cow", "a horse"], box_threshold=0.05):
        try:
            # Convert OpenCV BGR to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Ensure prompt is a list
            if isinstance(prompt, str):
                text_prompts = [p.strip() for p in prompt.split(",")]
            else:
                text_prompts = prompt

            # Encode image + text
            inputs = self.processor(text=text_prompts, images=image_rgb, return_tensors="pt").to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Proper target size tensor
            height, width = image_rgb.shape[:2]
            target_sizes = torch.tensor([[height, width]], device=self.device)

            # Post-process detections
            processed = self.processor.post_process_object_detection(
                outputs, threshold=box_threshold, target_sizes=target_sizes
            )[0]

            boxes = processed["boxes"].cpu().numpy()
            scores = processed["scores"].cpu().numpy()
            label_indices = processed["labels"].cpu().numpy()

            # Map numeric labels → actual text prompts
            mapped_labels = []
            for idx in label_indices:
                if idx < len(text_prompts):
                    lbl = text_prompts[idx]
                    # Clean label text for readability
                    lbl = lbl.replace("a ", "").replace("an ", "").strip()
                    mapped_labels.append(lbl)
                else:
                    mapped_labels.append(str(idx))

            if len(boxes) == 0:
                print("[INFO] OwlViT found no objects above threshold.")
                return processed

            # Print detections
            print(f"[INFO] OwlViT detected {len(boxes)} objects:")
            for label, score, box in zip(mapped_labels, scores, boxes):
                print(f"  - {label} ({score:.2f}) at {box.astype(int).tolist()}")

            # Replace numeric labels in processed output
            processed["labels"] = mapped_labels

            return processed  # Return processed result

        except Exception as e:
            print(f"[ERROR] OwlViT prompt detection failed: {e}")
            return {"boxes": [], "scores": [], "labels": []}

    def draw(self, image_bgr, results):
        """Draw vivid OwlViT detections."""
        if results is None or len(results.get("boxes", [])) == 0:
            return image_bgr

        img = image_bgr.copy()
        boxes = results["boxes"].cpu().numpy() if torch.is_tensor(results["boxes"]) else np.array(results["boxes"])
        scores = results["scores"].cpu().numpy() if torch.is_tensor(results["scores"]) else np.array(results["scores"])
        labels = results["labels"]

        for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
            if score < 0.05:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label_str = f"{label} {score:.2f}"

            color_map = {
                "person": (0, 255, 255),
                "pedestrian": (0, 255, 255),
                "animal": (0, 165, 255),
                "deer": (0, 0, 255),
                "vehicle": (0, 255, 0),
                "car": (0, 255, 0),
                "truck": (0, 128, 255),
                "cone": (255, 255, 0),
                "stop sign": (255, 0, 0),
                "dog": (255, 128, 0),
                "cow": (255, 64, 64),
                "horse": (255, 128, 128)
            }
            color = color_map.get(str(label).lower(), (255, 255, 255))

            # Draw vivid box and filled label
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label_str, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        return img
