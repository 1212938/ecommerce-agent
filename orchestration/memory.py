"""
记忆系统 — 短期记忆 + 长期记忆

短期记忆 (ShortTermMemory):
  - 维护最近 N 轮对话 (滑动窗口)
  - 超过阈值时自动摘要旧消息
  - 提供 LLM 上下文构建

长期记忆 (LongTermMemory):
  - 从对话中提取关键事实 (用户偏好、历史决策)
  - 向量化存储 (复用 FAISS + BGE)
  - 检索时按相关性召回

MemoryManager:
  - 统一管理短期 + 长期记忆
  - 按 session_id 隔离不同会话
  - 提供 build_context() 构建 LLM 上下文
"""
import os
import json
import time
import pickle
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from config.settings import settings


@dataclass
class Message:
    """单条消息"""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)  # 存储 intent, agent 等元信息


class ShortTermMemory:
    """
    短期记忆 — 滑动窗口 + 递归摘要

    工作原理:
    1. 保留最近 memory_window 条消息 (默认 10 条)
    2. 当消息数超过 memory_summary_threshold (默认 6) 时
       将较早的消息压缩为摘要，只保留最近窗口
    3. 摘要由 LLM 生成，提取关键信息
    4. 递归摘要: 当已有摘要 + 新摘要超过长度上限时，
       将两者合并后再次压缩，确保摘要不会无限增长
    """

    MAX_SUMMARY_LENGTH = 500  # 摘要最大字符数（超过则递归压缩）

    def __init__(self, window_size: int = None, summary_threshold: int = None, llm=None):
        self.window_size = window_size or settings.memory_window
        self.summary_threshold = summary_threshold or settings.memory_summary_threshold
        self.llm = llm
        self.messages: List[Message] = []
        self.summary: str = ""  # 历史摘要

    def add(self, role: str, content: str, **metadata):
        """添加一条消息"""
        msg = Message(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        self._maybe_summarize()

    def _maybe_summarize(self):
        """当消息数超过阈值时触发递归摘要"""
        if len(self.messages) <= self.summary_threshold:
            return

        # 需要被摘要的旧消息 (保留最近 window_size 条)
        old_messages = self.messages[:-self.window_size]
        if not old_messages:
            return

        # 生成新摘要
        new_summary = self._generate_summary(old_messages)

        # 递归合并: 已有摘要 + 新摘要 → 压缩为统一摘要
        if self.summary:
            combined = f"{self.summary}\n{new_summary}"
            # 超过长度上限时递归压缩
            if len(combined) > self.MAX_SUMMARY_LENGTH:
                self.summary = self._recursive_compress(combined)
            else:
                self.summary = combined
        else:
            self.summary = new_summary

        # 只保留最近窗口的消息
        self.messages = self.messages[-self.window_size:]

    def _generate_summary(self, messages: List[Message]) -> str:
        """使用 LLM 摘要旧消息"""
        if not self.llm:
            # 无 LLM 时使用简单截断
            return self._simple_summary(messages)

        conversation = "\n".join(
            f"{'用户' if m.role == 'user' else '助手'}: {m.content[:200]}"
            for m in messages
        )

        prompt = f"""请将以下对话历史压缩为简洁摘要，保留关键信息（用户意图、偏好、重要结论）。

{conversation}

摘要 (200字以内):"""

        try:
            result = self.llm.invoke(prompt)
            return result.content.strip()
        except Exception:
            return self._simple_summary(messages)

    def _recursive_compress(self, combined_summary: str) -> str:
        """
        递归压缩: 将过长的合并摘要再次用 LLM 压缩

        当已有摘要 + 新摘要合并后超过 MAX_SUMMARY_LENGTH 时调用。
        确保摘要长度始终受控，不会无限增长。
        """
        if not self.llm:
            # 无 LLM 时简单截断
            return combined_summary[:self.MAX_SUMMARY_LENGTH]

        prompt = f"""以下是一段过长的对话历史摘要，请将其进一步压缩为简洁摘要，
保留最重要的信息（用户偏好、关键决策、核心需求），删除冗余细节。

原始摘要:
{combined_summary}

压缩后摘要 ({self.MAX_SUMMARY_LENGTH // 2}字以内):"""

        try:
            result = self.llm.invoke(prompt)
            compressed = result.content.strip()
            # 确保压缩后的摘要确实更短
            if len(compressed) < len(combined_summary):
                return compressed
            return combined_summary[:self.MAX_SUMMARY_LENGTH]
        except Exception:
            return combined_summary[:self.MAX_SUMMARY_LENGTH]

    def _simple_summary(self, messages: List[Message]) -> str:
        """无 LLM 时的简单摘要"""
        user_msgs = [m.content[:50] for m in messages if m.role == "user"]
        return f"之前用户问了: {'; '.join(user_msgs)}"

    def get_recent(self, n: int = None) -> List[Message]:
        """获取最近 N 条消息"""
        n = n or self.window_size
        return self.messages[-n:]

    def build_context_messages(self) -> List[dict]:
        """
        构建 LLM 可用的上下文消息列表

        格式: [{"role": "system", "content": "摘要..."}, {"role": "user", "content": "..."}, ...]
        """
        result = []

        # 如果有摘要，作为 system 消息注入
        if self.summary:
            result.append({
                "role": "system",
                "content": f"[对话历史摘要]\n{self.summary}"
            })

        # 添加最近窗口的消息
        for msg in self.messages:
            result.append({
                "role": msg.role,
                "content": msg.content,
            })

        return result

    def get_user_history(self) -> List[str]:
        """获取所有用户消息（用于长期记忆提取）"""
        return [m.content for m in self.messages if m.role == "user"]

    def clear(self):
        """清空记忆"""
        self.messages.clear()
        self.summary = ""


@dataclass
class MemoryItem:
    """长期记忆条目"""
    content: str  # 记忆内容
    category: str  # "preference" | "fact" | "decision"
    session_id: str
    timestamp: float = field(default_factory=time.time)
    embedding: list = field(default_factory=list)


class LongTermMemory:
    """
    长期记忆 — 向量化存储与检索

    工作原理:
    1. 从对话中提取关键事实 (用户偏好、历史决策)
    2. 使用 BGE 模型向量化
    3. 存储到磁盘 (JSON + FAISS)
    4. 检索时按语义相似度召回 Top-K

    注意: 需要传入 shared_embedder (SentenceTransformer) 实例
    """

    def __init__(self, store_path: str = None, embedder=None, llm=None):
        self.store_path = store_path or settings.memory_store_path
        self.embedder = embedder
        self.llm = llm
        self.memories: List[MemoryItem] = []
        self._faiss_index = None
        self._loaded = False

        # 确保存储目录存在
        os.makedirs(self.store_path, exist_ok=True)

    def _ensure_loaded(self):
        """懒加载持久化记忆"""
        if self._loaded:
            return
        self._load()
        self._loaded = True

    def _load(self):
        """从磁盘加载记忆"""
        index_file = os.path.join(self.store_path, "memories.json")
        if os.path.exists(index_file):
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.memories = [
                    MemoryItem(
                        content=item["content"],
                        category=item.get("category", "fact"),
                        session_id=item.get("session_id", ""),
                        timestamp=item.get("timestamp", time.time()),
                    )
                    for item in data
                ]
            except Exception:
                self.memories = []

        # 加载 FAISS 索引
        faiss_file = os.path.join(self.store_path, "memories.index")
        if os.path.exists(faiss_file) and self.embedder:
            try:
                import faiss
                import numpy as np
                self._faiss_index = faiss.read_index(faiss_file)
            except Exception:
                pass

    def _save(self):
        """持久化记忆到磁盘"""
        index_file = os.path.join(self.store_path, "memories.json")
        try:
            data = [
                {
                    "content": m.content,
                    "category": m.category,
                    "session_id": m.session_id,
                    "timestamp": m.timestamp,
                }
                for m in self.memories
            ]
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _build_faiss_index(self):
        """构建/重建 FAISS 索引"""
        if not self.embedder or not self.memories:
            return

        try:
            import faiss
            import numpy as np

            texts = [m.content for m in self.memories]
            embeddings = self.embedder.encode(texts, normalize_embeddings=True).astype(np.float32)

            dim = embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(embeddings)

            faiss_file = os.path.join(self.store_path, "memories.index")
            faiss.write_index(self._faiss_index, faiss_file)
        except Exception:
            pass

    def extract_and_store(self, user_input: str, assistant_response: str, session_id: str):
        """
        从对话中提取关键信息并存储

        使用 LLM 判断是否有值得记住的信息 (偏好、事实、决策)
        """
        if not self.llm:
            return

        prompt = f"""分析以下对话，提取值得长期记住的关键信息（用户偏好、重要事实、购买决策等）。
如果没有值得记住的信息，回复 "NONE"。
否则用 JSON 格式回复: {{"items": [{{"content": "...", "category": "preference|fact|decision"}}]}}

用户: {user_input}
助手: {assistant_response}

提取结果:"""

        try:
            result = self.llm.invoke(prompt).content.strip()
            if result.upper().startswith("NONE"):
                return

            # 清理 markdown
            if result.startswith("```"):
                result = result.split("\n", 1)[-1]
                result = result.rsplit("```", 1)[0].strip()

            parsed = json.loads(result)
            for item in parsed.get("items", []):
                self._add_memory(
                    content=item["content"],
                    category=item.get("category", "fact"),
                    session_id=session_id,
                )
        except Exception:
            pass

    def _add_memory(self, content: str, category: str, session_id: str):
        """添加一条长期记忆"""
        # 去重: 检查是否已有相似记忆
        existing = self.search(content, k=1)
        if existing and existing[0].get("score", 0) > 0.92:
            return  # 已有高度相似的记忆，跳过

        item = MemoryItem(
            content=content,
            category=category,
            session_id=session_id,
        )

        # 生成 embedding
        if self.embedder:
            emb = self.embedder.encode([content], normalize_embeddings=True)
            item.embedding = emb[0].tolist()

        self.memories.append(item)
        self._save()
        self._build_faiss_index()

    def search(self, query: str, k: int = 3) -> List[dict]:
        """检索相关记忆"""
        self._ensure_loaded()

        if not self.memories:
            return []

        # 有 FAISS 索引时使用向量检索
        if self._faiss_index is not None and self.embedder:
            try:
                import numpy as np
                query_emb = self.embedder.encode(
                    [query], normalize_embeddings=True
                ).astype(np.float32)
                scores, indices = self._faiss_index.search(query_emb, min(k, len(self.memories)))

                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or idx >= len(self.memories):
                        continue
                    mem = self.memories[idx]
                    results.append({
                        "content": mem.content,
                        "category": mem.category,
                        "score": float(score),
                    })
                return results
            except Exception:
                pass

        # 降级: 关键词匹配
        results = []
        query_lower = query.lower()
        for mem in self.memories:
            score = sum(1 for kw in query_lower.split() if kw in mem.content.lower())
            if score > 0:
                results.append({
                    "content": mem.content,
                    "category": mem.category,
                    "score": score / max(len(query_lower.split()), 1),
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def get_context_for_query(self, query: str, k: int = 3) -> str:
        """为查询构建长期记忆上下文"""
        memories = self.search(query, k=k)
        if not memories:
            return ""

        lines = ["[用户历史记忆]"]
        for m in memories:
            lines.append(f"- {m['content']} ({m['category']})")
        return "\n".join(lines)


class MemoryManager:
    """
    记忆管理器 — 统一管理短期 + 长期记忆

    按 session_id 隔离不同会话的记忆。
    """

    def __init__(self, llm=None, embedder=None):
        self.llm = llm
        self.embedder = embedder
        self._short_term: Dict[str, ShortTermMemory] = {}
        self._long_term = LongTermMemory(
            embedder=embedder,
            llm=llm,
        ) if settings.memory_long_term_enabled else None

    def get_short_term(self, session_id: str) -> ShortTermMemory:
        """获取或创建会话的短期记忆"""
        if session_id not in self._short_term:
            self._short_term[session_id] = ShortTermMemory(
                llm=self.llm,
            )
        return self._short_term[session_id]

    def add_message(self, session_id: str, role: str, content: str, **metadata):
        """添加消息到短期记忆"""
        stm = self.get_short_term(session_id)
        stm.add(role, content, **metadata)

        # 如果是完整的 user-assistant 对，提取长期记忆
        if role == "assistant" and self._long_term:
            recent = stm.get_recent(2)
            if len(recent) >= 2 and recent[0].role == "user":
                self._long_term.extract_and_store(
                    user_input=recent[0].content,
                    assistant_response=recent[1].content,
                    session_id=session_id,
                )

    def build_context(self, session_id: str, current_query: str = "") -> List[dict]:
        """
        构建 LLM 上下文消息列表

        包含: 长期记忆 + 短期记忆摘要 + 最近对话
        """
        messages = []

        # 长期记忆
        if self._long_term and current_query:
            long_term_ctx = self._long_term.get_context_for_query(current_query)
            if long_term_ctx:
                messages.append({
                    "role": "system",
                    "content": long_term_ctx,
                })

        # 短期记忆 (摘要 + 最近窗口)
        stm = self.get_short_term(session_id)
        messages.extend(stm.build_context_messages())

        return messages

    def get_history(self, session_id: str) -> List[dict]:
        """获取会话历史 (用于 API 返回)"""
        stm = self.get_short_term(session_id)
        return [
            {"role": m.role, "content": m.content, "metadata": m.metadata}
            for m in stm.messages
        ]

    def clear_session(self, session_id: str):
        """清除会话记忆"""
        if session_id in self._short_term:
            self._short_term[session_id].clear()
            del self._short_term[session_id]

    def get_stats(self) -> dict:
        """获取记忆系统统计"""
        return {
            "active_sessions": len(self._short_term),
            "long_term_memories": len(self._long_term.memories) if self._long_term else 0,
        }
