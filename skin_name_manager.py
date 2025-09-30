import json
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - faiss 安装失败时的兜底
    faiss = None

from sentence_transformers import SentenceTransformer
from update_skin_name import update_skin_list

class SkinNameManager:
    """
    管理 CS2 皮肤名称的查询和匹配 (v4)。
    使用向量数据库和文本向量化模型（如 BERT / CLIP）进行相似度匹配。
    """
    
    # 1. 内置磨损词典
    WEAR_CONDITIONS: Dict[str, List[str]] = {
        "(Factory New)": ["fn", "factory new", "崭新出厂", "崭新"],
        "(Minimal Wear)": ["mw", "minimal wear", "略有磨损", "略磨"],
        "(Field-Tested)": ["ft", "field tested", "久经沙场", "久经"],
        "(Well-Worn)": ["ww", "well worn", "破烂不堪", "破烂"],
        "(Battle-Scarred)": ["bs", "battle scarred", "战痕累累", "战痕"],
    }
    
    def __init__(
        self,
        skin_list_filepath: str,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device: Optional[str] = None,
        rebuild_index: bool = True,
        embedding_cache_path: str = "skin_embeddings.npy",
        name_mapping_path: str = "skin_name_mapping.json",
        refresh_data: bool = False,
    ):
        """
        初始化管理器并从文件加载皮肤列表。

        :param skin_list_filepath: 指向 skin_list.txt 文件的路径。
        :param embedding_model: 用于向量化的模型名称，默认使用轻量级 BERT 变体。
        :param device: 指定句向量模型运行的设备（如 "cpu"、"cuda"）。None 时自动选择。
        :param rebuild_index: 是否在初始化时构建 / 载入向量索引与缓存。
        :param embedding_cache_path: 向量缓存文件的路径（.npy）。
        :param name_mapping_path: 名称映射文件（别名 -> 英文名）的路径。
        :param refresh_data: 是否在初始化时重新抓取数据并刷新缓存。
        """
        self.embedding_model_name = embedding_model
        self.embedding_device = device
        self.embedding_cache_path = Path(embedding_cache_path)
        self.name_mapping_path = Path(name_mapping_path)
        self._embedding_model: Optional[SentenceTransformer] = None
        self._vector_index = None
        self._embeddings: Optional[np.ndarray] = None
        self._vector_dim: Optional[int] = None
        self.alias_to_canonical: Dict[str, str] = {}

        if refresh_data:
            print("刷新标志开启：开始更新皮肤列表与向量缓存...")
            update_skin_list()

        self.skin_names: List[str] = []
        try:
            print(f"正在从 {skin_list_filepath} 加载皮肤名称列表...")
            with open(skin_list_filepath, 'r', encoding='utf-8') as f:
                self.skin_names = [line.strip() for line in f if line.strip()]
            
            if not self.skin_names:
                print("警告：皮肤列表为空或加载失败。")
            else:
                print(f"成功加载 {len(self.skin_names)} 个皮肤名称。")
                
        except FileNotFoundError:
            print(f"错误：找不到皮肤列表文件 at '{skin_list_filepath}'。")
            self.skin_names = []
        
        self._load_alias_mapping()

        if rebuild_index:
            self._build_vector_index(force_rebuild=refresh_data)

    def _load_alias_mapping(self) -> None:
        alias_map: Dict[str, str] = {}
        mapping_path = self.name_mapping_path

        if mapping_path.exists():
            try:
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if isinstance(data.get("alias_to_en"), dict):
                        alias_map = {
                            str(alias): str(canonical)
                            for alias, canonical in data["alias_to_en"].items()
                        }
                    else:
                        alias_map = {
                            str(alias): str(canonical)
                            for alias, canonical in data.items()
                        }
            except Exception as exc:
                print(f"读取名称映射文件失败，将退回默认映射。原因: {exc}")

        if not alias_map:
            alias_map = {name: name for name in self.skin_names}
        else:
            for name in self.skin_names:
                alias_map.setdefault(name, name)

        self.alias_to_canonical = alias_map

    def _get_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            print(f"正在加载向量化模型: {self.embedding_model_name} ...")
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.embedding_device,
            )
            print("向量化模型加载完成。")
        return self._embedding_model

    def _ensure_embeddings(self, force_rebuild: bool = False) -> None:
        if not self.skin_names:
            print("警告：皮肤列表为空，无法准备向量缓存。")
            self._embeddings = None
            self._vector_dim = None
            return

        if self._embeddings is not None and not force_rebuild:
            return

        cache_path = self.embedding_cache_path

        if not force_rebuild and cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=False)
                if cached.ndim == 2 and cached.shape[0] == len(self.skin_names):
                    self._embeddings = cached.astype(np.float32, copy=False)
                    self._vector_dim = self._embeddings.shape[1]
                    print(
                        f"已从缓存加载向量矩阵: {cache_path} -> 形状 {self._embeddings.shape}"
                    )
                    return
                else:
                    print(
                        "检测到缓存文件与当前皮肤列表长度不匹配，将重新计算嵌入。"
                    )
            except Exception as exc:  # pragma: no cover - 缓存损坏处理
                print(f"读取缓存 {cache_path} 失败，将重新生成向量。原因: {exc}")

        model = self._get_embedding_model()
        print("正在生成皮肤名称向量并写入缓存...")
        embeddings = model.encode(
            self.skin_names,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        print(f"向量缓存已保存至: {cache_path}，形状 {embeddings.shape}")

        self._embeddings = embeddings
        self._vector_dim = embeddings.shape[1]

    def _build_vector_index(self, force_rebuild: bool = False) -> None:
        if not self.skin_names:
            print("警告：皮肤列表为空，无法构建向量索引。")
            self._vector_index = None
            self._embeddings = None
            return

        self._ensure_embeddings(force_rebuild=force_rebuild)

        if self._embeddings is None:
            print("错误：未能准备好向量数据，无法构建索引。")
            self._vector_index = None
            return

        if faiss is not None:
            index = faiss.IndexFlatIP(self._vector_dim)
            index.add(self._embeddings)
            self._vector_index = index
            print(
                f"向量索引就绪（Faiss），维度 {self._vector_dim}，共 {len(self.skin_names)} 条记录。"
            )
        else:
            self._vector_index = None
            print(
                "警告：faiss-cpu 未安装，将退回使用 NumPy 逐一计算相似度，性能可能较低。"
            )

    def _vector_search(self, query_text: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if not self.skin_names:
            return []

        self._ensure_embeddings()

        model = self._get_embedding_model()
        query_embedding = model.encode(
            [query_text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        if faiss is not None and self._vector_index is None:
            self._build_vector_index()

        if self._vector_index is not None:
            scores, indices = self._vector_index.search(query_embedding, top_k)
            return list(zip(indices[0].tolist(), scores[0].tolist()))

        if self._embeddings is None:
            return []

        similarities = np.dot(self._embeddings, query_embedding[0])
        top_indices = np.argsort(-similarities)[:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def find_best_match(self, user_query: str, score_cutoff: int = 50) -> Optional[str]:
        """
        (v4) 使用向量数据库进行相似度匹配，根据用户输入找到最相似的皮肤名称。

        :param user_query: 用户输入的模糊查询，例如 "ak redline ft" 或 "汰换"。
        :param score_cutoff: 匹配度阈值 (0-100)。向量相似度在 [-1, 1]，内部会转换为 [0, 1] 比较。
        :return: 最佳匹配的 market_hash_name 字符串，如果找不到则返回 None。
        """
        if not self.skin_names:
            print("错误：皮肤列表未加载，无法进行匹配。")
            return None

        normalized_query = (user_query or "").strip()
        if not normalized_query:
            print("错误：用户查询为空，无法进行匹配。")
            return None

        normalized_query_lower = normalized_query.lower()
        print(f"处理查询 '{user_query}' -> 规范化结果: '{normalized_query_lower}'")

        similarity_threshold = score_cutoff / 100.0
        search_text_candidates = []
        if normalized_query_lower:
            search_text_candidates.append(normalized_query_lower)
        if normalized_query not in search_text_candidates:
            search_text_candidates.append(normalized_query)

        best_idx: Optional[int] = None
        best_score: float = -1.0
        used_query: Optional[str] = None

        for text in search_text_candidates:
            results = self._vector_search(text, top_k=3)
            if not results:
                continue
            idx, score = results[0]
            print(f"向量检索 '{text}' -> 候选 '{self.skin_names[idx]}'，相似度 {score:.4f}")
            if score > best_score:
                best_idx = idx
                best_score = score
                used_query = text
            if score >= similarity_threshold:
                break

        if best_idx is None:
            print(f"未能为 '{user_query}' 找到合适的皮肤匹配。")
            return None

        if best_score < similarity_threshold:
            print(
                f"最优候选 '{self.skin_names[best_idx]}' 相似度 {best_score:.4f} 低于阈值 {similarity_threshold:.2f}，返回 None。"
            )
            return None

        best_skin_base_name = self.skin_names[best_idx]
        print(
            f"向量检索成功：使用查询 '{used_query}' 找到 '{best_skin_base_name}'，相似度 {best_score:.4f}"
        )

        canonical_name = self.alias_to_canonical.get(best_skin_base_name, best_skin_base_name)
        if canonical_name != best_skin_base_name:
            print(f"归一别名映射 -> 返回英文名称: '{canonical_name}'")

        return canonical_name


if __name__ == "__main__":
    manager = SkinNameManager(
        "skin_list.txt",
        embedding_cache_path="skin_embeddings.npy",
        name_mapping_path="skin_name_mapping.json",
        refresh_data=True,
    )
    sample_queries = [
        "ak47红线",
        "印花",
        "M4A1-S Nightmare MW",
    ]
    for query in sample_queries:
        result = manager.find_best_match(query)
        print(f"[DEBUG] 查询 '{query}' -> 匹配结果: {result}")