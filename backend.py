import os
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import torch

from input import TextNormalizer

try:
    from FlagEmbedding import BGEM3FlagModel
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False


@dataclass
class RecommendationResult:
    doc_id: str
    image_path: Optional[str]
    final_score: float
    retrieval_score: float
    personalization_score: float
    alpha: float


class ThompsonSamplingBandit:
    """alpha 선택을 위한 Thompson Sampling bandit"""

    def __init__(self, alpha_candidates: List[float], config_path: Path):
        self.alpha_candidates = alpha_candidates
        self.config_path = config_path
        self.alpha_stats: Dict[str, Dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for a in self.alpha_candidates:
                    self.alpha_stats[str(a)] = data.get(str(a), {"successes": 1, "failures": 1})
            except Exception:
                self.alpha_stats = {}
        for a in self.alpha_candidates:
            self.alpha_stats.setdefault(str(a), {"successes": 1, "failures": 1})

    def _save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.alpha_stats, f, ensure_ascii=False, indent=2)

    def select(self) -> float:
        samples = {}
        for a in self.alpha_candidates:
            stats = self.alpha_stats[str(a)]
            samples[a] = np.random.beta(stats["successes"], stats["failures"])
        return max(samples, key=samples.get)

    def update(self, alpha: float, reward: float) -> None:
        stats = self.alpha_stats.setdefault(str(alpha), {"successes": 1, "failures": 1})
        if reward > 0:
            stats["successes"] += reward
        else:
            stats["failures"] += 1
        self._save()


class HybridRecommender:
    """텍스트+개인화 하이브리드 추천 파이프라인"""

    def __init__(
        self,
        artifacts_dir: Optional[Path] = None,
        alpha_config: Optional[Path] = None,
        model_name: str = "dragonkue/BGE-m3-ko",
        images_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        self.base_dir = Path(artifacts_dir or Path.cwd())
        self.alpha_config = Path(alpha_config or (self.base_dir / "alpha_config.json"))
        self.images_dir = Path(images_dir or Path.cwd() / "pictures")
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not FLAG_EMBEDDING_AVAILABLE:
            raise ImportError("FlagEmbedding 패키지가 필요합니다. pip install FlagEmbedding")

        self.normalizer = TextNormalizer()
        self.embedding_model = BGEM3FlagModel(
            self.model_name,
            use_fp16=(self.device == "cuda" and torch.cuda.is_available()),
            normalize_to_unit=True,
            device=self.device,
        )

        self.faiss_index = self._load_faiss_index(self.base_dir / "faiss.index")
        self.row_id_to_doc_id = self._load_id_map(self.base_dir / "id_map.parquet")
        (
            self.lightfm_model,
            self.user2idx,
            self.item2idx,
            self.item_features,
        ) = self._load_lightfm(self.base_dir / "lightfm.pkl")

        self.bandit = ThompsonSamplingBandit([0.8, 0.65, 0.5, 0.4], self.alpha_config)

    @staticmethod
    def _load_faiss_index(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"FAISS 인덱스를 찾을 수 없습니다: {path}")
        return faiss.read_index(str(path))

    @staticmethod
    def _load_id_map(path: Path) -> Dict[int, str]:
        if not path.exists():
            raise FileNotFoundError(f"id_map.parquet 파일이 필요합니다: {path}")
        df = pd.read_parquet(path)
        # 기본 컬럼명(row_id, doc_id) 보장
        row_col = "row_id" if "row_id" in df.columns else df.columns[0]
        doc_col = "doc_id" if "doc_id" in df.columns else df.columns[1]
        mapping = dict(zip(df[row_col].astype(int), df[doc_col].astype(str)))
        return mapping

    @staticmethod
    def _load_lightfm(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"LightFM 모델 파일이 필요합니다: {path}")
        with open(path, "rb") as f:
            artifact = pickle.load(f)
        model = artifact["model"]
        dataset = artifact["dataset"]
        item_features = artifact.get("item_features")
        user2idx, _, item2idx, _ = dataset.mapping()
        return model, user2idx, item2idx, item_features

    # -------------------- 임베딩 & 검색 --------------------
    def _normalize_and_embed(self, text: str) -> np.ndarray:
        normalized = self.normalizer.normalize(
            text,
            classify_intent=False,
            use_similarity=True,
            similarity_threshold=0.8,
            convert_slang=True,
        )
        prefixed = f"query: {normalized}"
        output = self.embedding_model.encode(
            [prefixed],
            batch_size=1,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output["dense_vecs"].astype(np.float32)

    def _search_faiss(self, query_embedding: np.ndarray, top_k: int = 20) -> Tuple[List[str], np.ndarray]:
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        doc_ids: List[str] = []
        scores = []
        for row_id, score in zip(indices[0], distances[0]):
            if row_id == -1:
                continue
            doc_id = self.row_id_to_doc_id.get(int(row_id))
            if doc_id is not None:
                doc_ids.append(str(doc_id))
                scores.append(score)
        return doc_ids, np.array(scores, dtype=np.float32)

    # -------------------- LightFM 개인화 --------------------
    def _to_item_id(self, doc_id: str) -> str:
        doc_id = str(doc_id).strip()
        digits = "".join(ch for ch in doc_id if ch.isdigit())
        if digits:
            return f"P{digits}"
        return doc_id

    def _personalization_scores(self, user_id: str, doc_ids: List[str]) -> np.ndarray:
        user_idx = self.user2idx.get(user_id)
        if user_idx is None:
            return np.ones(len(doc_ids)) * 0.5

        item_indices = []
        for doc_id in doc_ids:
            item_idx = self.item2idx.get(self._to_item_id(doc_id))
            item_indices.append(item_idx)

        valid_positions = [i for i, idx in enumerate(item_indices) if idx is not None]
        if not valid_positions:
            return np.ones(len(doc_ids)) * 0.5

        valid_indices = [item_indices[i] for i in valid_positions]
        if self.item_features is not None:
            features = self.item_features[valid_indices]
            raw_scores = self.lightfm_model.predict(user_idx, valid_indices, item_features=features)
        else:
            raw_scores = self.lightfm_model.predict(user_idx, valid_indices)

        scores = np.ones(len(doc_ids)) * 0.3
        for pos, sc in zip(valid_positions, raw_scores):
            scores[pos] = float(sc)

        return 1 / (1 + np.exp(-scores * 0.5))

    # -------------------- 헬퍼 --------------------
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val < 1e-8:
            return np.ones_like(scores) * 0.5
        return (scores - min_val) / (max_val - min_val)

    def _find_image_path(self, doc_id: str) -> Optional[str]:
        candidates = [
            self.images_dir / f"{doc_id}.png",
            self.images_dir / f"{doc_id}.jpg",
            self.images_dir / f"{doc_id}.jpeg",
            self.images_dir / f"P{doc_id}.png",
            self.images_dir / f"P{doc_id}.jpg",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return None

    # -------------------- 공개 메소드 --------------------
    def recommend(self, query: str, user_id: str, top_k: int = 8) -> List[RecommendationResult]:
        if not query or not query.strip():
            return []

        alpha = self.bandit.select()
        query_embedding = self._normalize_and_embed(query)
        # 확인용
        doc_ids, retrieval_raw = self._search_faiss(query_embedding, top_k=20)
        print(f"[FAISS] query={query!r}, "f"top_ids={doc_ids[:8]}",flush=True)
        if not doc_ids:
            return []

        retrieval_norm = self._normalize_scores(retrieval_raw)
        personalization = self._personalization_scores(user_id, doc_ids)
        personalization = self._normalize_scores(personalization)

        final_scores = alpha * retrieval_norm + (1 - alpha) * personalization

        order = np.argsort(final_scores)[::-1][:top_k]
        results: List[RecommendationResult] = []
        for idx in order:
            doc_id = doc_ids[idx]
            results.append(
                RecommendationResult(
                    doc_id=str(doc_id),
                    image_path=self._find_image_path(doc_id),
                    final_score=float(final_scores[idx]),
                    retrieval_score=float(retrieval_norm[idx]),
                    personalization_score=float(personalization[idx]),
                    alpha=alpha,
                )
            )
        return results

    def reward(self, alpha: float, reward: float) -> None:
        self.bandit.update(alpha, reward)
