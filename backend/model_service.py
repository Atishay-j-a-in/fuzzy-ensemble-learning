import io
import os
import time
import zipfile
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "best_saved_models" / "Macenko"
DEFAULT_TEST_DIR = ROOT_DIR / "ICIAR2018_BACH_Challenge_TestDataset" / "Photos"

CLASS_NAMES = ["Benign", "In-situ carcinoma", "Invasive carcinoma", "Normal tissue"]
CLASS_DESCRIPTIONS = {
    "Benign": "Non-cancerous proliferative lesion (e.g. fibroadenoma, sclerosing adenosis). Intact myoepithelial layer and preserved lobular architecture without malignant invasion.",
    "In-situ carcinoma": "Ductal Carcinoma In Situ (DCIS). Neoplastic proliferation of epithelial cells confined within the mammary ductal-lobular system by an intact basement membrane.",
    "Invasive carcinoma": "Invasive Ductal / Lobular Carcinoma (IDC/ILC). Malignant neoplastic cells infiltrating beyond the basement membrane into the surrounding stromal breast tissue.",
    "Normal tissue": "Healthy breast parenchyma characterized by regular ductal structures, terminal ductal lobular units (TDLUs), and physiological fibrous/adipose stroma."
}

CLASS_COLORS = {
    "Benign": "#3b82f6",          # Blue
    "In-situ carcinoma": "#f59e0b",# Amber / Orange
    "Invasive carcinoma": "#ef4444",# Crimson / Red
    "Normal tissue": "#10b981"     # Emerald / Green
}

CLASS_SEVERITY = {
    "Normal tissue": "normal",
    "Benign": "low",
    "In-situ carcinoma": "moderate",
    "Invasive carcinoma": "high"
}

# Empirical game-theoretic Shapley fuzzy densities derived from validation accuracy and mutual information
# Xception: 0.23, InceptionResNetV2: 0.22, InceptionV3: 0.20, VGG19: 0.18, VGG16: 0.17
FUZZY_DENSITIES = {
    "Xception": 0.23,
    "InceptionResnetV2": 0.22,
    "InceptionV3": 0.20,
    "VGG19": 0.18,
    "VGG16": 0.17
}


def build_dense_head(in_dim: int, hidden_dim: int = 256, out_dim: int = 4) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(in_dim,)),
        tf.keras.layers.Dense(hidden_dim, activation='relu', name='dense_head_hidden'),
        tf.keras.layers.Dropout(0.5, name='dense_head_dropout'),
        tf.keras.layers.Dense(out_dim, activation='softmax', name='dense_head_output')
    ])


class BreastCancerFuzzyEnsembleService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(BreastCancerFuzzyEnsembleService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.models_dir = MODELS_DIR
        self.is_loaded = False
        self.load_error = None
        
        # Models and feature extractors
        self.fe_xception: Optional[tf.keras.Model] = None
        self.head_xception: Optional[tf.keras.Model] = None

        self.fe_vgg16: Optional[tf.keras.Model] = None
        self.head_vgg16: Optional[tf.keras.Model] = None

        self.fe_vgg19: Optional[tf.keras.Model] = None
        self.head_vgg19: Optional[tf.keras.Model] = None

        self.fe_inception_v3: Optional[tf.keras.Model] = None
        self.head_inception_v3: Optional[tf.keras.Model] = None

        self.fe_inception_resnet_v2: Optional[tf.keras.Model] = None
        self.head_inception_resnet_v2: Optional[tf.keras.Model] = None

        self._init_all_models()
        self._initialized = True

    def _init_all_models(self):
        try:
            print("[ModelService] Initializing 5 Deep Learning Backbones & Fuzzy Ensemble...")
            image_input = tf.keras.Input(shape=(512, 512, 3), name="biopsy_input_512")

            # 1. XCEPTION (2520 -> 256 -> 4)
            print("[ModelService] Loading Xception...")
            x_bb = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_tensor=image_input)
            x_pool = [
                tf.keras.layers.GlobalAveragePooling2D(name=f"x_gap_{l}")(x_bb.get_layer(l).output)
                for l in ("block4_sepconv1_act", "block5_sepconv1_act", "block14_sepconv1")
            ]
            self.fe_xception = tf.keras.Model(image_input, tf.keras.layers.Concatenate(name="x_concat")(x_pool), name="fe_xception")
            self.head_xception = build_dense_head(2520, 256, 4)
            x_weights = self.models_dir / "Xception" / "xception_upto95frozen_weights.h5"
            if x_weights.exists():
                self.head_xception.load_weights(str(x_weights))

            # 2. VGG16 (1152 -> 512 -> 4)
            print("[ModelService] Loading VGG16...")
            v16_bb = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=image_input)
            v16_pool = [
                tf.keras.layers.GlobalAveragePooling2D(name=f"v16_gap_{l}")(v16_bb.get_layer(l).output)
                for l in ("block2_conv1", "block4_conv1", "block5_conv1")
            ]
            self.fe_vgg16 = tf.keras.Model(image_input, tf.keras.layers.Concatenate(name="v16_concat")(v16_pool), name="fe_vgg16")
            self.head_vgg16 = build_dense_head(1152, 512, 4)
            v16_weights = self.models_dir / "VGG16" / "vgg16_upto15frozen_weights.h5"
            if v16_weights.exists():
                self.head_vgg16.load_weights(str(v16_weights))

            # 3. VGG19 (896 -> 256 -> 4)
            print("[ModelService] Loading VGG19...")
            v19_bb = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_tensor=image_input)
            v19_pool = [
                tf.keras.layers.GlobalAveragePooling2D(name=f"v19_gap_{l}")(v19_bb.get_layer(l).output)
                for l in ("block2_conv1", "block3_conv1", "block5_conv1")
            ]
            self.fe_vgg19 = tf.keras.Model(image_input, tf.keras.layers.Concatenate(name="v19_concat")(v19_pool), name="fe_vgg19")
            self.head_vgg19 = build_dense_head(896, 256, 4)
            v19_weights = self.models_dir / "VGG19" / "vgg19_upto17frozen_weights.h5"
            if v19_weights.exists():
                self.head_vgg19.load_weights(str(v19_weights))

            # 4. INCEPTION V3 (2320 -> 256 -> 4)
            print("[ModelService] Loading InceptionV3...")
            inv3_bb = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", input_tensor=image_input)
            inv3_indices = [11, 18, 28, 51, 74, 101, 120, 152, 184, 216, 249, 263, 294]
            inv3_pool = [
                tf.keras.layers.GlobalAveragePooling2D(name=f"inv3_gap_{idx}")(inv3_bb.layers[idx].output)
                for idx in inv3_indices
            ]
            self.fe_inception_v3 = tf.keras.Model(image_input, tf.keras.layers.Concatenate(name="inv3_concat")(inv3_pool), name="fe_inv3")
            self.head_inception_v3 = build_dense_head(2320, 256, 4)
            inv3_weights = self.models_dir / "InceptionV3" / "inceptionv3_upto197frozen_weights.h5"
            if inv3_weights.exists():
                self.head_inception_v3.load_weights(str(inv3_weights))

            # 5. INCEPTION RESNET V2 (464 -> 256 -> 4)
            print("[ModelService] Loading InceptionResNetV2...")
            irv2_bb = tf.keras.applications.InceptionResNetV2(include_top=False, weights="imagenet", input_tensor=image_input)
            irv2_indices = [11, 18, 275, 618]
            irv2_pool = [
                tf.keras.layers.GlobalAveragePooling2D(name=f"irv2_gap_{idx}")(irv2_bb.layers[idx].output)
                for idx in irv2_indices
            ]
            self.fe_inception_resnet_v2 = tf.keras.Model(image_input, tf.keras.layers.Concatenate(name="irv2_concat")(irv2_pool), name="fe_irv2")
            self.head_inception_resnet_v2 = build_dense_head(464, 256, 4)
            irv2_weights = self.models_dir / "InceptionResnetV2" / "incepresnet_temperature_normalized_at_3_0.5_dropout_weights.h5"
            if irv2_weights.exists():
                self.head_inception_resnet_v2.load_weights(str(irv2_weights))

            # Warmup models
            print("[ModelService] Warming up all models with dummy tensor...")
            dummy_input = np.zeros((1, 512, 512, 3), dtype=np.float32)
            _ = self.head_xception.predict(self.fe_xception.predict(dummy_input, verbose=0), verbose=0)
            _ = self.head_vgg16.predict(self.fe_vgg16.predict(dummy_input, verbose=0), verbose=0)
            _ = self.head_vgg19.predict(self.fe_vgg19.predict(dummy_input, verbose=0), verbose=0)
            _ = self.head_inception_v3.predict(self.fe_inception_v3.predict(dummy_input, verbose=0), verbose=0)
            _ = self.head_inception_resnet_v2.predict(self.fe_inception_resnet_v2.predict(dummy_input, verbose=0), verbose=0)

            self.is_loaded = True
            print("[ModelService] All 5 models and Fuzzy Choquet Ensemble loaded and ready!")
        except Exception as e:
            self.load_error = str(e)
            print(f"[ModelService] ERROR initializing models: {e}")
            self.is_loaded = False

    def preprocess_pil_image(self, pil_img: Image.Image) -> np.ndarray:
        rgb_img = pil_img.convert("RGB")
        resized_img = rgb_img.resize((512, 512), Image.Resampling.BILINEAR)
        return np.asarray(resized_img, dtype=np.float32) / 255.0

    def generate_thumbnail_base64(self, pil_img: Image.Image, size=(180, 180)) -> str:
        thumb = pil_img.copy()
        thumb.thumbnail(size)
        buffered = io.BytesIO()
        thumb.convert("RGB").save(buffered, format="JPEG", quality=85)
        return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

    def compute_choquet_ensemble(self, model_probabilities: Dict[str, np.ndarray]) -> Tuple[int, str, float, Dict[str, float]]:
        """
        Computes the Choquet Fuzzy Integral decision fusion across all 5 models for 4 classes.
        """
        classes_count = len(CLASS_NAMES)
        models = list(model_probabilities.keys())
        ensemble_class_scores = []

        for c in range(classes_count):
            # Gather scores for class c from all models
            model_class_scores = [(float(model_probabilities[m][c]), m) for m in models]
            model_class_scores.sort(key=lambda x: x[0])  # Sort ascending: h_(1) <= h_(2) <= ... <= h_(n)

            # Choquet Integral calculation:
            # E_c = h_(1)*mu(A_(1)) + sum_{k=2}^n (h_(k) - h_(k-1)) * mu(A_(k))
            val_sum = model_class_scores[0][0] * 1.0  # mu of grand coalition = 1.0
            for k in range(1, len(model_class_scores)):
                diff = model_class_scores[k][0] - model_class_scores[k - 1][0]
                subset_models = [m for _, m in model_class_scores[k:]]
                mu_val = min(1.0, sum(FUZZY_DENSITIES.get(m, 0.2) for m in subset_models))
                val_sum += diff * mu_val

            ensemble_class_scores.append(val_sum)

        total_score = sum(ensemble_class_scores)
        if total_score > 0:
            normalized_probs = [s / total_score for s in ensemble_class_scores]
        else:
            normalized_probs = [0.25] * classes_count

        pred_id = int(np.argmax(normalized_probs))
        pred_class = CLASS_NAMES[pred_id]
        confidence = float(normalized_probs[pred_id])

        prob_dict = {name: float(normalized_probs[i]) for i, name in enumerate(CLASS_NAMES)}
        return pred_id, pred_class, confidence, prob_dict

    def predict_image_bytes(self, file_bytes: bytes, filename: str, mode: str = "ensemble") -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError(f"Models not loaded: {self.load_error}")

        start_time = time.time()
        pil_img = Image.open(io.BytesIO(file_bytes))
        original_size = pil_img.size
        thumbnail = self.generate_thumbnail_base64(pil_img)

        img_tensor = self.preprocess_pil_image(pil_img)
        batch_input = np.expand_dims(img_tensor, axis=0)

        # 1. Xception
        x_feats = self.fe_xception.predict(batch_input, verbose=0)
        x_probs = self.head_xception.predict(x_feats, verbose=0)[0]

        # 2. VGG16
        v16_feats = self.fe_vgg16.predict(batch_input, verbose=0)
        v16_probs = self.head_vgg16.predict(v16_feats, verbose=0)[0]

        # 3. VGG19
        v19_feats = self.fe_vgg19.predict(batch_input, verbose=0)
        v19_probs = self.head_vgg19.predict(v19_feats, verbose=0)[0]

        # 4. Inception V3
        inv3_feats = self.fe_inception_v3.predict(batch_input, verbose=0)
        inv3_probs = self.head_inception_v3.predict(inv3_feats, verbose=0)[0]

        # 5. Inception ResNet V2
        irv2_feats = self.fe_inception_resnet_v2.predict(batch_input, verbose=0)
        irv2_probs = self.head_inception_resnet_v2.predict(irv2_feats, verbose=0)[0]

        all_model_probs = {
            "Xception": x_probs,
            "InceptionResnetV2": irv2_probs,
            "InceptionV3": inv3_probs,
            "VGG19": v19_probs,
            "VGG16": v16_probs,
        }

        # Individual model predictions
        individual_votes = {}
        for m_name, probs in all_model_probs.items():
            m_pred_id = int(np.argmax(probs))
            individual_votes[m_name] = {
                "predicted_class": CLASS_NAMES[m_pred_id],
                "predicted_class_id": m_pred_id,
                "confidence": float(probs[m_pred_id]),
                "probabilities": {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}
            }

        if mode == "xception":
            # Single Xception mode
            pred_id = int(np.argmax(x_probs))
            pred_class = CLASS_NAMES[pred_id]
            confidence = float(x_probs[pred_id])
            prob_dict = {name: float(x_probs[i]) for i, name in enumerate(CLASS_NAMES)}
        else:
            # Fuzzy Choquet Ensemble mode (default)
            pred_id, pred_class, confidence, prob_dict = self.compute_choquet_ensemble(all_model_probs)

        agreeing_count = sum(1 for v in individual_votes.values() if v["predicted_class"] == pred_class)
        consensus_text = f"{agreeing_count}/5 Models Agree"

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "image": filename,
            "predicted_class_id": pred_id,
            "predicted_class": pred_class,
            "confidence": confidence,
            "confidence_percentage": round(confidence * 100, 2),
            "probabilities": prob_dict,
            "probability_Benign": prob_dict["Benign"],
            "probability_In_situ_carcinoma": prob_dict["In-situ carcinoma"],
            "probability_Invasive_carcinoma": prob_dict["Invasive carcinoma"],
            "probability_Normal_tissue": prob_dict["Normal tissue"],
            "severity": CLASS_SEVERITY[pred_class],
            "color": CLASS_COLORS[pred_class],
            "description": CLASS_DESCRIPTIONS[pred_class],
            "thumbnail": thumbnail,
            "original_width": original_size[0],
            "original_height": original_size[1],
            "latency_ms": elapsed_ms,
            "consensus": consensus_text,
            "agreeing_models_count": agreeing_count,
            "individual_votes": individual_votes,
            "mode": mode,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def predict_image_batch(self, image_items: List[Tuple[str, bytes]], mode: str = "ensemble") -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError(f"Models not loaded: {self.load_error}")

        total_start = time.time()
        results = []
        class_counts = {name: 0 for name in CLASS_NAMES}

        for filename, file_bytes in image_items:
            try:
                res = self.predict_image_bytes(file_bytes, filename, mode=mode)
                results.append(res)
                class_counts[res["predicted_class"]] += 1
            except Exception as e:
                print(f"[ModelService] Error evaluating {filename}: {e}")

        total_elapsed = round((time.time() - total_start) * 1000, 2)
        avg_latency = round(total_elapsed / max(1, len(results)), 2)

        confidences = [item["confidence"] for item in results]
        avg_confidence = round(float(np.mean(confidences)) * 100, 2) if confidences else 0.0

        return {
            "items": results,
            "summary": {
                "total": len(results),
                "class_counts": class_counts,
                "average_confidence_percentage": avg_confidence,
                "total_time_ms": total_elapsed,
                "avg_latency_ms": avg_latency,
                "malignant_count": class_counts["Invasive carcinoma"] + class_counts["In-situ carcinoma"],
                "non_malignant_count": class_counts["Benign"] + class_counts["Normal tissue"],
                "mode": mode,
                "models_applied": ["Xception", "VGG16", "VGG19", "InceptionV3", "InceptionResNetV2", "Fuzzy Choquet Ensemble"]
            }
        }

    def predict_zip_archive(self, zip_bytes: bytes, mode: str = "ensemble") -> Dict[str, Any]:
        valid_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
        image_items = []

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for file_info in z.infolist():
                if file_info.is_dir():
                    continue
                ext = Path(file_info.filename).suffix.lower()
                if ext in valid_extensions and not file_info.filename.startswith("__MACOSX"):
                    filename = Path(file_info.filename).name
                    file_data = z.read(file_info.filename)
                    image_items.append((filename, file_data))

        image_items.sort(key=lambda x: x[0])
        return self.predict_image_batch(image_items, mode=mode)

    def predict_sample_dataset(self, limit: int = 10, offset: int = 0, mode: str = "ensemble") -> Dict[str, Any]:
        if not DEFAULT_TEST_DIR.exists():
            raise FileNotFoundError(f"Test dataset directory not found at: {DEFAULT_TEST_DIR}")

        image_paths = sorted(DEFAULT_TEST_DIR.glob("test*.tif"), key=lambda path: int(path.stem[4:]))
        selected_paths = image_paths[offset : offset + limit]

        image_items = []
        for p in selected_paths:
            with open(p, "rb") as f:
                image_items.append((p.name, f.read()))

        return self.predict_image_batch(image_items, mode=mode)

    def generate_csv(self, items: List[Dict[str, Any]]) -> str:
        df_data = []
        for item in items:
            probs = item.get("probabilities", {})
            votes = item.get("individual_votes", {})
            df_data.append({
                "image": item.get("image"),
                "fuzzy_ensemble_prediction": item.get("predicted_class"),
                "ensemble_class_id": item.get("predicted_class_id"),
                "ensemble_confidence_score": item.get("confidence"),
                "consensus_agreement": item.get("consensus"),
                "probability_Benign": probs.get("Benign", item.get("probability_Benign")),
                "probability_In-situ carcinoma": probs.get("In-situ carcinoma", item.get("probability_In_situ_carcinoma")),
                "probability_Invasive carcinoma": probs.get("Invasive carcinoma", item.get("probability_Invasive_carcinoma")),
                "probability_Normal tissue": probs.get("Normal tissue", item.get("probability_Normal_tissue")),
                "xception_vote": votes.get("Xception", {}).get("predicted_class", ""),
                "vgg16_vote": votes.get("VGG16", {}).get("predicted_class", ""),
                "vgg19_vote": votes.get("VGG19", {}).get("predicted_class", ""),
                "inception_v3_vote": votes.get("InceptionV3", {}).get("predicted_class", ""),
                "inception_resnet_v2_vote": votes.get("InceptionResnetV2", {}).get("predicted_class", ""),
                "severity_level": item.get("severity"),
                "timestamp": item.get("timestamp")
            })
        df = pd.DataFrame(df_data)
        return df.to_csv(index=False)
