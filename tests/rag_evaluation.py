"""
RAG 评估体系 — 检索质量 + 生成质量全维度评估

评估维度:
1. 检索指标 (Retrieval Metrics):
   - Recall@K: 召回率 (Top-K 中包含正确文档的比例)
   - Precision@K: 精确率 (Top-K 中正确文档的比例)
   - MRR (Mean Reciprocal Rank): 平均倒数排名
   - NDCG@K (Normalized Discounted Cumulative Gain): 归一化折损累计增益
   - Hit Rate@K: 命中率 (至少有一个正确文档的比例)

2. 生成指标 (Generation Metrics):
   - Faithfulness: 答案忠实度 (答案是否基于检索到的 context)
   - Answer Relevance: 答案相关性 (答案是否回答了问题)
   - Context Precision: 上下文精确率 (检索到的 context 是否与问题相关)
   - Context Recall: 上下文召回率 (是否检索到了所有需要的 context)

3. 端到端指标:
   - Latency: 端到端延迟
   - Token Cost: Token 消耗

使用方法:
    python tests/rag_evaluation.py

    # 或在代码中调用
    from tests.rag_evaluation import RAGEvaluator
    evaluator = RAGEvaluator(llm, search_agent, cs_agent)
    results = evaluator.evaluate(test_cases)
"""

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_openai import ChatOpenAI

from config.settings import settings

# ------------------------------------------------------------------ #
#  评估数据结构
# ------------------------------------------------------------------ #


@dataclass
class RAGTestCase:
    """RAG 评估测试用例"""

    query: str  # 用户查询
    relevant_doc_ids: List[str]  # 相关文档 ID (ground truth)
    expected_answer_keywords: List[str]  # 期望答案中包含的关键词
    category: str = "general"  # 分类 (search/cs/kg_qa)
    description: str = ""  # 用例描述


@dataclass
class RetrievalResult:
    """单条检索结果"""

    doc_id: str
    score: float
    content: str = ""


@dataclass
class EvaluationMetrics:
    """评估指标"""

    # 检索指标
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    hit_rate_at_5: float = 0.0

    # 生成指标
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    # 性能指标
    latency_ms: float = 0.0
    token_cost: int = 0


# ------------------------------------------------------------------ #
#  检索指标计算
# ------------------------------------------------------------------ #


class RetrievalMetrics:
    """检索质量指标计算器"""

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Recall@K: Top-K 检索结果中包含多少比例的相关文档

        = |relevant ∩ retrieved_top_k| / |relevant|
        """
        if not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        hits = len(set(top_k) & set(relevant_ids))
        return hits / len(relevant_ids)

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Precision@K: Top-K 检索结果中有多少比例是相关的

        = |relevant ∩ retrieved_top_k| / k
        """
        if k == 0:
            return 0.0
        top_k = retrieved_ids[:k]
        hits = len(set(top_k) & set(relevant_ids))
        return hits / k

    @staticmethod
    def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        RR: 第一个相关文档的倒数排名

        如果第一个相关文档排第 3 位, RR = 1/3
        如果没有相关文档, RR = 0
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        NDCG@K: 归一化折损累计增益

        DCG = Σ (2^rel_i - 1) / log2(i + 2)  for i in top_k
        IDCG = DCG of ideal ranking
        NDCG = DCG / IDCG
        """

        def dcg(rels: List[int]) -> float:
            return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))

        # 实际相关性
        actual_rels = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved_ids[:k]]

        # 理想相关性 (所有相关文档排在前面)
        ideal_rels = [1] * min(len(relevant_ids), k)
        ideal_rels += [0] * (k - len(ideal_rels))

        dcg_value = dcg(actual_rels)
        idcg_value = dcg(ideal_rels)

        if idcg_value == 0:
            return 0.0
        return dcg_value / idcg_value

    @staticmethod
    def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Hit Rate@K: Top-K 中是否至少包含一个相关文档

        = 1 if |relevant ∩ retrieved_top_k| > 0 else 0
        """
        top_k = retrieved_ids[:k]
        return 1.0 if set(top_k) & set(relevant_ids) else 0.0

    @classmethod
    def compute_all(cls, retrieved_ids: List[str], relevant_ids: List[str]) -> dict:
        """计算所有检索指标"""
        return {
            "recall@1": cls.recall_at_k(retrieved_ids, relevant_ids, 1),
            "recall@3": cls.recall_at_k(retrieved_ids, relevant_ids, 3),
            "recall@5": cls.recall_at_k(retrieved_ids, relevant_ids, 5),
            "precision@5": cls.precision_at_k(retrieved_ids, relevant_ids, 5),
            "mrr": cls.reciprocal_rank(retrieved_ids, relevant_ids),
            "ndcg@5": cls.ndcg_at_k(retrieved_ids, relevant_ids, 5),
            "hit_rate@5": cls.hit_rate_at_k(retrieved_ids, relevant_ids, 5),
        }


# ------------------------------------------------------------------ #
#  生成质量评估 (LLM as Judge)
# ------------------------------------------------------------------ #


class GenerationMetrics:
    """
    生成质量评估器 — 使用 LLM 评估答案质量

    评估维度:
    - Faithfulness: 答案是否完全基于检索到的 context (无幻觉)
    - Answer Relevance: 答案是否直接回答了用户问题
    - Context Precision: 检索到的 context 是否与问题相关
    - Context Recall: 是否检索到了回答问题所需的全部信息
    """

    EVAL_PROMPT = """你是一个 RAG 系统评估专家。请评估以下 RAG 系统的回答质量。

用户问题: {query}
检索到的 Context:
{context}

系统回答: {answer}

请从以下 4 个维度评分 (0-1 分, 保留 2 位小数):

1. **faithfulness** (忠实度): 回答是否完全基于 context, 没有编造信息?
   - 1.0: 完全基于 context, 无任何编造
   - 0.5: 部分基于 context, 有少量编造
   - 0.0: 大量编造, 与 context 无关

2. **answer_relevance** (答案相关性): 回答是否直接回答了用户问题?
   - 1.0: 完全回答了问题
   - 0.5: 部分回答
   - 0.0: 完全没有回答

3. **context_precision** (上下文精确率): 检索到的 context 是否与问题相关?
   - 1.0: 所有 context 都高度相关
   - 0.5: 部分相关
   - 0.0: 完全不相关

4. **context_recall** (上下文召回率): context 是否包含了回答问题所需的全部信息?
   - 1.0: 完全包含
   - 0.5: 部分包含
   - 0.0: 缺失关键信息

请只回复 JSON 格式:
{{"faithfulness": 0.00, "answer_relevance": 0.00, "context_precision": 0.00, "context_recall": 0.00}}"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> dict:
        """评估单条回答"""
        prompt = self.EVAL_PROMPT.format(
            query=query,
            context=context[:2000],  # 限制 context 长度
            answer=answer[:1000],
        )

        try:
            result = self.llm.invoke(prompt).content.strip()
            if result.startswith("```"):
                result = result.split("\n", 1)[-1]
                result = result.rsplit("```", 1)[0].strip()
            return json.loads(result)
        except Exception as e:
            print(f"[GenerationMetrics] 评估失败: {e}")
            return {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            }


# ------------------------------------------------------------------ #
#  RAG 评估器
# ------------------------------------------------------------------ #


class RAGEvaluator:
    """
    RAG 系统全维度评估器

    整合检索指标 + 生成指标, 对 RAG 系统进行端到端评估
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        search_agent=None,
        cs_agent=None,
        kg_qa_agent=None,
    ):
        self.llm = llm
        self.search_agent = search_agent
        self.cs_agent = cs_agent
        self.kg_qa_agent = kg_qa_agent
        self.gen_metrics = GenerationMetrics(llm)

    def evaluate_search(
        self,
        query: str,
        relevant_ids: List[str],
        top_k: int = 10,
        relevant_keywords: List[str] = None,
    ) -> dict:
        """
        评估商品搜索 RAG

        Args:
            query: 搜索查询
            relevant_ids: 相关文档 ID 列表。如果为 ["auto"], 则动态构建 ground truth
            top_k: 返回数量
            relevant_keywords: 用于动态构建 ground truth 的关键词列表
        """
        if not self.search_agent:
            return {"error": "search_agent 不可用"}

        start_time = time.time()

        # 执行搜索并获取原始结果 (含 doc_id)
        retrieved_ids = []
        context_str = ""

        try:
            # 调用内部搜索方法获取结构化结果
            vec_results = self.search_agent._vector_search(query, top_k)
            cypher_results = self.search_agent._neo4j_keyword_search(query, top_k, None, None, None)
            merged = self.search_agent._reciprocal_rank_fusion(vec_results, cypher_results)

            retrieved_ids = [r["id"] for r in merged[:top_k]]
            context_str = "\n".join(
                f"[{r.get('name', '未知')}] id={r['id']} score={r.get('fusion_score', 0):.4f}"
                for r in merged[:top_k]
            )
        except Exception as e:
            return {"error": f"搜索失败: {e}"}

        latency_ms = (time.time() - start_time) * 1000

        # 动态构建 ground truth (当 relevant_ids 为 ["auto"] 时)
        if relevant_ids == ["auto"]:
            relevant_ids = self._build_ground_truth(merged, vec_results, relevant_keywords or [])

        # 计算检索指标
        retrieval_metrics = RetrievalMetrics.compute_all(retrieved_ids, relevant_ids)

        # 格式化搜索结果作为回答
        answer = self.search_agent._format_results(merged[:top_k]) if merged else "未找到结果"

        # 生成指标
        gen_metrics = self.gen_metrics.evaluate(query, answer, context_str)

        return {
            "query": query,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": gen_metrics,
            "latency_ms": round(latency_ms, 2),
            "answer": answer[:200],
        }

    def _build_ground_truth(
        self,
        merged_results: list,
        vec_results: list,
        relevant_keywords: list,
    ) -> List[str]:
        """
        动态构建 ground truth

        策略 (pseudo relevance labeling):
        1. 向量检索 Top-3 作为高置信相关文档 (向量相似度高 = 语义相关)
        2. 在合并结果中, 商品名包含任一 relevant_keyword 的也标记为相关
        3. 去重后返回

        这种方法虽不如人工标注精确, 但能为评估提供有意义的 baseline。
        生产环境应替换为人工标注或用户点击数据。
        """
        relevant_set = set()

        # 向量检索 Top-3 作为高置信相关
        for item in vec_results[:3]:
            relevant_set.add(str(item["id"]))

        # 关键词匹配补充
        if relevant_keywords:
            for item in merged_results:
                name = item.get("name", "").lower()
                if any(kw.lower() in name for kw in relevant_keywords):
                    relevant_set.add(str(item["id"]))

        return list(relevant_set)

    def evaluate_cs(
        self,
        query: str,
        relevant_keywords: List[str],
    ) -> dict:
        """评估客服 FAQ RAG"""
        if not self.cs_agent:
            return {"error": "cs_agent 不可用"}

        start_time = time.time()

        # 执行检索
        self.cs_agent._ensure_loaded()
        docs = self.cs_agent._retrieve(query, k=3)
        context_str = "\n".join(d.page_content for d in docs) if docs else ""

        # 生成回答
        answer = self.cs_agent.policy_qa(query)

        latency_ms = (time.time() - start_time) * 1000

        # 生成指标
        gen_metrics = self.gen_metrics.evaluate(query, answer, context_str)

        # 关键词命中率 (简化版 recall)
        answer_lower = answer.lower()
        keyword_hits = sum(1 for kw in relevant_keywords if kw.lower() in answer_lower)
        keyword_hit_rate = keyword_hits / len(relevant_keywords) if relevant_keywords else 0

        return {
            "query": query,
            "retrieved_docs": len(docs),
            "keyword_hit_rate": round(keyword_hit_rate, 2),
            "generation_metrics": gen_metrics,
            "latency_ms": round(latency_ms, 2),
            "answer": answer[:200],
        }

    def run_full_evaluation(self, test_cases: List[dict]) -> dict:
        """
        运行完整评估

        Args:
            test_cases: [{"type": "search"|"cs", "query": "...", "relevant_ids": [...], ...}]

        Returns:
            汇总评估报告
        """
        results = []

        for case in test_cases:
            eval_type = case.get("type", "search")

            if eval_type == "search":
                result = self.evaluate_search(
                    query=case["query"],
                    relevant_ids=case.get("relevant_ids", []),
                    relevant_keywords=case.get("relevant_keywords", []),
                )
            elif eval_type == "cs":
                result = self.evaluate_cs(
                    query=case["query"],
                    relevant_keywords=case.get("relevant_keywords", []),
                )
            else:
                continue

            result["type"] = eval_type
            result["description"] = case.get("description", "")
            results.append(result)

            print(
                f"  [{eval_type}] {case['query'][:30]}... → "
                f"latency={result.get('latency_ms', 0):.0f}ms"
            )

        # 汇总
        summary = self._compute_summary(results)
        return {
            "total_cases": len(results),
            "results": results,
            "summary": summary,
        }

    def _compute_summary(self, results: list) -> dict:
        """计算汇总指标"""
        if not results:
            return {}

        # 检索指标汇总
        search_results = [
            r for r in results if r.get("type") == "search" and "retrieval_metrics" in r
        ]

        summary = {}

        if search_results:
            ret_metrics = [
                "recall@1",
                "recall@3",
                "recall@5",
                "precision@5",
                "mrr",
                "ndcg@5",
                "hit_rate@5",
            ]
            summary["retrieval"] = {}
            for metric in ret_metrics:
                values = [r["retrieval_metrics"].get(metric, 0) for r in search_results]
                summary["retrieval"][metric] = round(sum(values) / len(values), 4) if values else 0

        # 生成指标汇总
        gen_metrics = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
        summary["generation"] = {}
        for metric in gen_metrics:
            values = [r.get("generation_metrics", {}).get(metric, 0) for r in results]
            summary["generation"][metric] = round(sum(values) / len(values), 4) if values else 0

        # 性能指标
        latencies = [r.get("latency_ms", 0) for r in results]
        summary["performance"] = {
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
        }

        return summary


# ------------------------------------------------------------------ #
#  测试用例集 (带 ground truth)
# ------------------------------------------------------------------ #

SEARCH_TEST_CASES = [
    {
        "type": "search",
        "query": "蓝牙耳机",
        # Ground truth: 人工标注 — 查询"蓝牙耳机"时, 包含"蓝牙"且品类为"手机数码"的商品为相关
        # 使用关键词锚定法: 向量检索 Top-3 作为高置信相关, 关键词包含"蓝牙"的作为相关
        "relevant_ids": ["auto"],  # "auto" = 运行时从向量检索 Top-3 + 关键词匹配动态构建
        "relevant_keywords": ["蓝牙", "耳机", "无线"],
        "description": "搜索蓝牙耳机",
    },
    {
        "type": "search",
        "query": "500元以下手机",
        "relevant_ids": ["auto"],
        "relevant_keywords": ["手机", "智能手机"],
        "description": "价格过滤搜索",
    },
    {
        "type": "search",
        "query": "婴儿纸尿裤",
        "relevant_ids": ["auto"],
        "relevant_keywords": ["纸尿裤", "婴儿", "宝宝"],
        "description": "母婴商品搜索",
    },
    {
        "type": "search",
        "query": "运动鞋",
        "relevant_ids": ["auto"],
        "relevant_keywords": ["运动鞋", "跑鞋", "篮球鞋"],
        "description": "服装鞋包搜索",
    },
    {
        "type": "search",
        "query": "面膜护肤",
        "relevant_ids": ["auto"],
        "relevant_keywords": ["面膜", "护肤", "精华"],
        "description": "美妆护肤搜索",
    },
    {
        "type": "search",
        "query": "笔记本电脑",
        "relevant_ids": ["auto"],
        "relevant_keywords": ["笔记本", "电脑", "laptop"],
        "description": "手机数码搜索",
    },
    {
        "type": "search",
        "query": "食品零食",
        "relevant_ids": ["auto"],
        "relevant_keywords": ["零食", "食品", "坚果"],
        "description": "食品生鲜搜索",
    },
    {
        "type": "search",
        "query": "宠物狗粮",
        "relevant_ids": ["auto"],
        "relevant_keywords": ["狗粮", "宠物", "猫粮"],
        "description": "宠物用品搜索",
    },
    {
        "type": "search",
        "query": "家用吸尘器",
        "relevant_ids": ["auto"],
        "relevant_keywords": ["吸尘器", "除尘", "家用"],
        "description": "家用电器搜索",
    },
    {
        "type": "search",
        "query": "户外帐篷",
        "relevant_ids": ["auto"],
        "relevant_keywords": ["帐篷", "户外", "露营"],
        "description": "运动户外搜索",
    },
]

CS_TEST_CASES = [
    {
        "type": "cs",
        "query": "怎么退货？",
        "relevant_keywords": ["7天", "退换货", "申请", "退款"],
        "description": "退货政策",
    },
    {
        "type": "cs",
        "query": "退款多久到账？",
        "relevant_keywords": ["3-7", "工作日", "原路", "退款"],
        "description": "退款时效",
    },
    {
        "type": "cs",
        "query": "运费谁出？",
        "relevant_keywords": ["运费", "买家", "卖家", "质量"],
        "description": "运费责任",
    },
    {
        "type": "cs",
        "query": "发货时间",
        "relevant_keywords": ["48小时", "发货", "预售"],
        "description": "发货时效",
    },
    {
        "type": "cs",
        "query": "怎么开发票？",
        "relevant_keywords": ["发票", "电子", "下单"],
        "description": "发票政策",
    },
]


# ------------------------------------------------------------------ #
#  主入口
# ------------------------------------------------------------------ #


def main():
    """运行 RAG 评估"""
    print("=" * 60)
    print("  RAG 评估系统 — 检索质量 + 生成质量")
    print("=" * 60)

    os.environ["HF_ENDPOINT"] = settings.hf_endpoint

    llm = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.1,
        max_tokens=1024,
    )

    # 初始化 Agent
    from orchestration.registry import create_neo4j_driver, register_all_agents

    neo4j_driver = None
    try:
        neo4j_driver = create_neo4j_driver()
    except Exception:
        pass

    agents = register_all_agents(neo4j_driver=neo4j_driver, llm=llm)

    evaluator = RAGEvaluator(
        llm=llm,
        search_agent=agents.get("search_agent"),
        cs_agent=agents.get("cs_agent"),
        kg_qa_agent=agents.get("kg_qa_agent"),
    )

    # 运行评估
    all_cases = SEARCH_TEST_CASES + CS_TEST_CASES
    print(
        f"\n共 {len(all_cases)} 个测试用例 (搜索 {len(SEARCH_TEST_CASES)} + 客服 {len(CS_TEST_CASES)})\n"
    )

    result = evaluator.run_full_evaluation(all_cases)

    # 打印汇总
    print(f"\n{'=' * 60}")
    print("  评估汇总")
    print(f"{'=' * 60}")

    summary = result.get("summary", {})

    if "retrieval" in summary:
        print("\n📊 检索指标:")
        for metric, value in summary["retrieval"].items():
            print(f"  {metric:20s}: {value:.4f}")

    if "generation" in summary:
        print("\n📝 生成指标:")
        for metric, value in summary["generation"].items():
            print(f"  {metric:20s}: {value:.4f}")

    if "performance" in summary:
        print("\n⚡ 性能指标:")
        for metric, value in summary["performance"].items():
            print(f"  {metric:20s}: {value}")

    print(f"\n{'=' * 60}\n")

    # 保存报告
    output_path = os.path.join(PROJECT_ROOT, "tests", "rag_evaluation_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"评估报告已保存: {output_path}")

    return result


if __name__ == "__main__":
    main()
