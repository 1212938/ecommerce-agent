"""
推荐 Agent — 个性化商品推荐引擎

结合三种推荐策略：
1. 图结构推荐 (Neo4j 分类/品牌协同关系)
2. LLM 推理增强 (基于用户画像和需求的推理)
3. Item-CF 协同过滤 (基于 MySQL 订单数据的物品协同过滤)

学习参考: llm-based-recommender 的 LangGraph 推荐流程
          ec_graph 的 GNN 实体嵌入 (可扩展)
"""

import json
from typing import Optional

from agents.tools.base import BaseAgentTool


class RecommendAgent(BaseAgentTool):
    """
    个性化商品推荐引擎

    流程：
        用户需求 + 用户画像
        → 图结构初筛 (Neo4j 协同查询)
        → Item-CF 补充 (MySQL 订单协同过滤)
        → LLM 重排序 (基于用户偏好推理)
        → 格式化推荐列表
    """

    name: str = "recommend_agent"
    description: str = (
        "商品推荐：根据用户偏好、需求描述和画像信息进行个性化推荐，"
        "结合知识图谱协同关系、Item-CF 协同过滤和 LLM 推理"
    )

    @staticmethod
    def _pretty_name(name: str) -> str:
        """
        美化合成商品名: '华为手机数码商品302' → '华为手机数码'

        样本数据集商品名是 品牌+品类+序号 的合成格式，序号对用户无意义，
        去掉后展示更自然（真实数据集的商品名不受影响）。
        """
        if not name:
            return name
        import re

        cleaned = re.sub(r"商品\d+$", "", name).strip()
        return cleaned or name

    def __init__(self, neo4j_driver, llm, db_config: dict = None):
        super().__init__()
        self.neo4j_driver = neo4j_driver
        self.llm = llm
        self.db_config = db_config  # MySQL 连接配置 (用于 Item-CF)

    # ------------------------------------------------------------------ #
    #  对外接口
    # ------------------------------------------------------------------ #

    def run(self, query: str, **kwargs) -> str:
        """同步执行推荐"""
        user_profile = kwargs.get("user_profile")
        top_k = kwargs.get("top_k", 5)
        return self.recommend(query, user_profile, top_k)

    async def arun(self, **kwargs) -> str:
        """异步执行"""
        return self.run(**kwargs)

    # ------------------------------------------------------------------ #
    #  核心推荐逻辑
    # ------------------------------------------------------------------ #

    def recommend(self, query: str, user_profile: Optional[dict] = None, top_k: int = 5) -> str:
        """
        根据用户偏好和当前上下文推荐商品

        Args:
            query: 用户需求描述
            user_profile: 用户画像（可选），如 {"偏好分类": "母婴", "预算": "100-200"}
            top_k: 推荐数量

        Returns:
            格式化的推荐结果字符串
        """
        user_profile = user_profile or {}

        # Step 1: 图结构推荐（基于分类/品牌的协同关系）
        graph_recs = self._graph_based_recommend(query, user_profile, top_k * 2)

        # Step 2: Item-CF 协同过滤补充（基于 MySQL 订单共现）
        cf_recs = self._item_cf_recommend(query, user_profile, top_k * 2)

        # 合并去重: 图推荐 + CF 推荐
        seen_ids = set()
        merged_recs = []
        for item in graph_recs + cf_recs:
            item_id = str(item.get("id", ""))
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                merged_recs.append(item)

        # 如果合并后无结果，降级到 Neo4j 宽泛查询或告知用户（严禁 LLM 编造）
        if not merged_recs:
            return self._llm_direct_recommend(query, user_profile, top_k)

        # Step 3: LLM 重排序（基于用户画像的推理增强）
        reranked = self._llm_rerank(query, merged_recs, user_profile, top_k)

        # Step 4: 格式化输出
        return self._format_recommendations(reranked[:top_k])

    def _graph_based_recommend(self, query: str, user_profile: dict, top_k: int) -> list:
        """
        基于知识图谱的协同推荐

        策略：
        1. 从用户画像提取偏好分类和品牌
        2. 在图中查找该分类/品牌下的商品
        3. 通过品牌-分类协同关系扩展候选
        """
        # 提取用户偏好
        preferred_category = user_profile.get("偏好分类") or user_profile.get("category")
        preferred_brand = user_profile.get("偏好品牌") or user_profile.get("brand")
        budget = user_profile.get("预算") or user_profile.get("budget")

        # 如果用户画像无明确分类，用 LLM 从 query 中提取
        if not preferred_category:
            preferred_category = self._extract_category(query)

        if not preferred_category and not preferred_brand:
            # 无明确偏好，返回热门商品
            return self._get_popular_products(top_k)

        cypher = """
        MATCH (p:SPU)-[:Belong]->(c:Category3)
        WHERE ($category IS NULL OR c.name CONTAINS $category)
        OPTIONAL MATCH (p)-[:Have]->(t:Trademark)
        WHERE ($brand IS NULL OR t.name CONTAINS $brand OR t.tm_name CONTAINS $brand)
        OPTIONAL MATCH (p)-[:Have]->(sku:SKU)
        RETURN p.name AS product, p.id AS id,
               c.name AS category,
               collect(DISTINCT t.name)[0..2] AS brands,
               collect(DISTINCT sku.price)[0..3] AS prices
        LIMIT $top_k
        """

        try:
            with self.neo4j_driver.session(default_transaction_timeout=10) as session:
                result = session.run(
                    cypher,
                    {
                        "category": preferred_category,
                        "brand": preferred_brand,
                        "top_k": top_k,
                    },
                )
                records = []
                for r in result:
                    brands = r.get("brands", []) or []
                    prices = r.get("prices", []) or []
                    record = {
                        "product": r["product"],
                        "id": str(r["id"]),
                        "category": r.get("category", ""),
                        "brands": brands,
                        "prices": prices,
                        # 默认取首个品牌/价格，保证展示时有真实信息
                        "brand": brands[0] if brands else "",
                        "price": prices[0] if prices else "",
                    }
                    # 价格过滤
                    if budget:
                        price_range = self._parse_budget(budget)
                        if price_range:
                            min_p, max_p = price_range
                            valid_prices = [
                                float(p)
                                for p in prices
                                if p and self._in_range(float(p), min_p, max_p)
                            ]
                            if not valid_prices:
                                continue
                            record["price"] = valid_prices[0]
                    records.append(record)
                return records
        except Exception as e:
            print(f"[RecommendAgent] 图查询失败: {e}")
            return []

    def _item_cf_recommend(self, query: str, user_profile: dict, top_k: int) -> list:
        """
        Item-CF 协同过滤推荐 — 基于 MySQL 订单数据的物品协同过滤

        原理:
        1. 从用户偏好分类中提取种子商品 (seed items)
        2. 查询 MySQL: 购买过种子商品的用户还购买了什么
        3. 按共现频率排序 (co-occurrence count)
        4. 归一化为相似度分数 (cosine normalization)

        这是经典的 "购买了该商品的用户还购买了" 推荐策略。
        """
        if not self.db_config:
            return []

        import pymysql

        preferred_category = user_profile.get("偏好分类") or user_profile.get("category")
        if not preferred_category:
            preferred_category = self._extract_category(query)
        if not preferred_category:
            return []

        try:
            conn = pymysql.connect(**self.db_config)
        except Exception as e:
            print(f"[RecommendAgent] Item-CF MySQL 连接失败: {e}")
            return []

        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                # Step 1: 查找该分类下的种子商品 (用户购买过的)
                cursor.execute(
                    """SELECT DISTINCT od.product_id, od.product_name
                       FROM order_detail od
                       WHERE od.product_name LIKE %s
                       LIMIT 10""",
                    (f"%{preferred_category}%",),
                )
                seed_products = cursor.fetchall()
                if not seed_products:
                    return []

                seed_ids = [p["product_id"] for p in seed_products]

                # Step 2: 找到购买过这些种子商品的用户
                placeholders = ",".join(["%s"] * len(seed_ids))
                cursor.execute(
                    f"""SELECT DISTINCT user_id FROM order_detail
                        WHERE product_id IN ({placeholders})""",
                    seed_ids,
                )
                user_ids = [row["user_id"] for row in cursor.fetchall()]
                if not user_ids:
                    return []

                # 限制用户数量避免查询过大
                user_ids = user_ids[:500]
                user_placeholders = ",".join(["%s"] * len(user_ids))

                # Step 3: 这些用户还购买了什么 (co-purchase)
                cursor.execute(
                    f"""SELECT product_id, product_name, COUNT(*) as co_count
                        FROM order_detail
                        WHERE user_id IN ({user_placeholders})
                          AND product_id NOT IN ({placeholders})
                        GROUP BY product_id, product_name
                        ORDER BY co_count DESC
                        LIMIT %s""",
                    user_ids + seed_ids + [top_k],
                )
                co_purchased = cursor.fetchall()

                # Step 4: 计算归一化相似度分数并构建推荐列表
                max_count = co_purchased[0]["co_count"] if co_purchased else 1
                results = []
                for item in co_purchased:
                    results.append(
                        {
                            "product": item["product_name"],
                            "id": str(item["product_id"]),
                            "category": preferred_category,
                            "score": round(item["co_count"] / max_count * 10, 1),
                            "reason": f"{item['co_count']}位相似用户共同购买",
                        }
                    )

                return results
        except Exception as e:
            print(f"[RecommendAgent] Item-CF 查询失败: {e}")
            return []
        finally:
            conn.close()

    def _llm_rerank(self, query: str, candidates: list, user_profile: dict, top_k: int) -> list:
        """
        LLM 重排序：基于用户画像对候选商品进行推理重排

        输入候选商品列表 → LLM 评估每个商品的匹配度 → 按匹配度排序
        """
        if not candidates:
            return []

        # 格式化候选商品
        candidates_text = json.dumps(candidates, ensure_ascii=False, indent=2)

        prompt = f"""你是一个电商推荐专家。请根据用户需求重新排序候选商品。

用户需求: {query}
用户画像: {json.dumps(user_profile, ensure_ascii=False)}

候选商品(JSON):
{candidates_text}

请根据用户需求和画像，对候选商品按推荐优先级排序。
返回 JSON 数组格式，每个元素包含:
  - "product": 商品名
  - "reason": 推荐理由（一句话）
  - "score": 匹配度评分(0-10)

只返回 JSON 数组，不要其他文字。"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # 清理 markdown 标记
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content
                content = content.rsplit("```", 1)[0] if "```" in content else content
                content = content.strip()

            reranked = json.loads(content)
            # 按 score 降序
            reranked.sort(key=lambda x: float(x.get("score", 0)), reverse=True)

            # 合并原始信息
            for item in reranked:
                for orig in candidates:
                    if orig.get("product") == item.get("product"):
                        item.setdefault("id", orig.get("id", ""))
                        item.setdefault("category", orig.get("category", ""))
                        item.setdefault("price", orig.get("price", ""))
                        break

            return reranked[:top_k]
        except Exception as e:
            print(f"[RecommendAgent] LLM 重排序失败: {e}")
            return candidates[:top_k]

    def _llm_direct_recommend(self, query: str, user_profile: dict, top_k: int) -> str:
        """
        图查询和 Item-CF 均无结果时的降级处理

        严禁让 LLM 自行编造商品推荐。
        尝试用 Neo4j 宽泛查询作为最后兜底，如果仍无结果则如实告知用户。
        """
        # 尝试 Neo4j 宽泛查询（不做分类过滤，直接按名称模糊匹配）
        if self.neo4j_driver:
            try:
                cypher = """
                MATCH (p:SPU)-[:Belong]->(c:Category3)
                OPTIONAL MATCH (p)-[:Have]->(sku:SKU)
                RETURN p.name AS product, p.id AS id,
                       c.name AS category,
                       collect(DISTINCT sku.price)[0..1] AS prices
                LIMIT $top_k
                """
                with self.neo4j_driver.session(default_transaction_timeout=10) as session:
                    result = session.run(cypher, {"top_k": top_k})
                    items = []
                    for r in result:
                        name = r["product"] or ""
                        category = r.get("category", "")
                        prices = r.get("prices", [])
                        price = prices[0] if prices else ""
                        line = f"**{name}**"
                        if category:
                            line += f" | 分类: {category}"
                        if price:
                            line += f" | 价格: ¥{price}"
                        items.append(line)

                    if items:
                        header = f"💡 为您推荐 {len(items)} 个热门商品：\n"
                        return header + "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))
            except Exception as e:
                print(f"[RecommendAgent] Neo4j 兜底查询失败: {e}")

        # 所有数据源均无结果，如实告知
        return (
            f"抱歉，根据您的需求「{query}」，暂未在商品库中找到匹配的商品。\n"
            "您可以尝试：\n"
            "1. 换一个关键词搜索（如「搜索 耳机」）\n"
            "2. 浏览热门品类（如母婴、户外、食品）\n"
            "3. 描述更具体的需求（如「预算100以内的零食」）"
        )

    # 商品库 15 个一级分类（用于分类提取约束，避免 LLM 自由发挥）
    CATEGORY_CANDIDATES = [
        "食品生鲜",
        "医药保健",
        "服装鞋包",
        "汽车用品",
        "母婴用品",
        "箱包配饰",
        "美妆护肤",
        "家居家装",
        "手机数码",
        "宠物用品",
        "运动户外",
        "家用电器",
        "珠宝首饰",
        "图书音像",
        "礼品鲜花",
    ]

    def _extract_category(self, query: str) -> Optional[str]:
        """从用户查询中提取商品分类（限定在商品库 15 个一级分类内）"""
        candidates = "、".join(self.CATEGORY_CANDIDATES)
        prompt = f"""从以下用户需求中提取最匹配的商品分类，必须从候选分类中选择一个，只回复分类名，不要其他文字。
如果需求与任何候选分类都不相关，回复 "unknown"。

候选分类: {candidates}

用户需求: {query}

分类:"""
        try:
            result = self.llm.invoke(prompt).content.strip()
            # 校验结果必须是候选分类之一
            for cat in self.CATEGORY_CANDIDATES:
                if cat in result:
                    return cat
            return None
        except Exception:
            return None

    def _get_popular_products(self, top_k: int) -> list:
        """获取热门商品（无明确偏好时的兜底）"""
        # 使用确定性排序替代 ORDER BY rand()，避免全表扫描
        cypher = """
        MATCH (p:SPU)-[:Have]->(sku:SKU)
        RETURN p.name AS product, p.id AS id, sku.price AS price
        ORDER BY p.id LIMIT $top_k
        """
        try:
            with self.neo4j_driver.session(default_transaction_timeout=10) as session:
                result = session.run(cypher, {"top_k": top_k})
                return [
                    {
                        "product": r["product"],
                        "id": str(r["id"]),
                        "price": str(r.get("price", "")),
                    }
                    for r in result
                ]
        except Exception:
            return []

    def _parse_budget(self, budget) -> tuple:
        """解析预算字符串，返回 (min, max) 元组"""
        if isinstance(budget, (int, float)):
            return (0, float(budget))
        if isinstance(budget, str):
            import re

            nums = re.findall(r"\d+\.?\d*", budget)
            if len(nums) >= 2:
                return (float(nums[0]), float(nums[1]))
            elif len(nums) == 1:
                return (0, float(nums[0]))
        return None

    def _in_range(self, price: float, min_p: float, max_p: float) -> bool:
        """检查价格是否在范围内"""
        return min_p <= price <= max_p

    def _format_recommendations(self, items: list) -> str:
        """格式化推荐列表"""
        if not items:
            return "暂无推荐商品，请稍后再试。"

        lines = [f"💡 为您推荐 {len(items)} 个商品：\n"]
        for i, item in enumerate(items, 1):
            name = self._pretty_name(item.get("product", item.get("name", "未知商品")))
            reason = item.get("reason", "")
            category = item.get("category", "")
            # 图查询返回的是 brands 列表，需兼容两种字段
            brand = item.get("brand", "")
            if not brand and item.get("brands"):
                brand = (
                    item["brands"][0]
                    if isinstance(item["brands"], list) and item["brands"]
                    else item["brands"]
                )
            price = item.get("price", "")
            score = item.get("score", "")

            line = f"{i}. **{name}**"
            if brand:
                line += f" | 品牌: {brand}"
            if category:
                line += f" | 分类: {category}"
            if price:
                line += f" | 价格: ¥{price}"
            if reason:
                line += f"\n   推荐理由: {reason}"
            elif score:
                line += f" | 匹配度: {score}/10"
            lines.append(line)

        return "\n".join(lines)
