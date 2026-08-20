"""
Agent 单元测试

对各个子 Agent 进行独立测试，验证核心功能正常
不依赖外部服务（Neo4j/MySQL）时使用 Mock

运行方法:
    cd /opt/ecommerce-agent
    python -m pytest tests/test_agents.py -v

    或直接运行:
    python tests/test_agents.py
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from unittest.mock import MagicMock, patch

from config.settings import settings
from orchestration.router import RouterAgent


# ------------------------------------------------------------------ #
#  Router Agent 测试
# ------------------------------------------------------------------ #

class TestRouterAgent:
    """路由器测试"""

    @pytest.fixture
    def router(self):
        """创建路由器（使用 Mock LLM）"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="chitchat")
        return RouterAgent(mock_llm)

    def test_keyword_route_order(self, router):
        """测试订单意图识别"""
        result = router.route("查询我的订单 ORD123456")
        assert result["intent"] == "order"
        assert result["agent"] == "order_agent"

    def test_keyword_route_customer_service(self, router):
        """测试客服意图识别"""
        result = router.route("怎么退货？")
        assert result["intent"] == "customer_service"
        assert result["agent"] == "cs_agent"

    def test_keyword_route_search(self, router):
        """测试搜索意图识别"""
        result = router.route("搜索蓝牙耳机")
        assert result["intent"] == "search"
        assert result["agent"] == "search_agent"

    def test_keyword_route_analytics(self, router):
        """测试数据分析意图识别"""
        result = router.route("最近销量排行")
        assert result["intent"] == "analytics"
        assert result["agent"] == "analytics_agent"

    def test_keyword_route_classify(self, router):
        """测试分类意图识别（直接分类请求，非问句式）"""
        # "属于什么类别" 是问句，会路由到 kg_qa（优先级更高）
        # classify 仅处理直接分类请求
        result = router.route("分类: 纸尿裤")
        assert result["intent"] == "classify"

    def test_keyword_route_kg_qa_question(self, router):
        """测试问句式分类查询路由到知识图谱"""
        result = router.route("纸尿裤属于什么类别")
        assert result["intent"] == "kg_qa"
        assert result["agent"] == "kg_qa_agent"

    def test_keyword_route_recommend(self, router):
        """测试推荐意图识别"""
        result = router.route("有什么推荐的")
        assert result["intent"] == "recommend"

    def test_keyword_route_chitchat(self, router):
        """测试闲聊意图识别"""
        result = router.route("你好")
        assert result["intent"] == "chitchat"

    def test_normalize_intent(self, router):
        """测试意图标准化"""
        assert router._normalize_intent("cs") == "customer_service"
        assert router._normalize_intent("qa") == "kg_qa"
        assert router._normalize_intent("unknown_intent") == "chitchat"


# ------------------------------------------------------------------ #
#  Classify Agent 测试（规则降级模式）
# ------------------------------------------------------------------ #

class TestClassifyAgent:
    """分类 Agent 测试（使用规则降级模式，不依赖模型文件）"""

    @pytest.fixture
    def agent(self):
        from agents.classify_agent import ClassifyAgent
        # 指定不存在的路径，触发规则降级
        return ClassifyAgent(
            model_path="/nonexistent/model",
            labels_path="/nonexistent/labels.txt",
        )

    def test_rule_based_classify_phone(self, agent):
        """测试规则分类 - 手机"""
        result = agent.classify_product("小米手机 红米 Note 12")
        assert "手机数码" in result

    def test_rule_based_classify_clothes(self, agent):
        """测试规则分类 - 服装"""
        result = agent.classify_product("纯棉T恤 短袖")
        assert "服装鞋包" in result

    def test_rule_based_classify_food(self, agent):
        """测试规则分类 - 食品"""
        result = agent.classify_product("进口牛肉 牛排")
        assert "食品生鲜" in result

    def test_preprocess(self, agent):
        """测试文本预处理"""
        # 全角转半角
        result = agent._preprocess("Ａｐｐｌｅ手机　　128GB")
        assert "Apple" in result or "apple" in result.lower()
        # 多余空白被清除
        assert "  " not in result

    def test_get_top_k(self, agent):
        """测试 Top-K 返回"""
        results = agent.get_top_k("手机", k=3)
        assert isinstance(results, list)
        assert len(results) >= 1


# ------------------------------------------------------------------ #
#  Order Agent 测试
# ------------------------------------------------------------------ #

class TestOrderAgent:
    """订单 Agent 测试"""

    @pytest.fixture
    def agent(self):
        from agents.order_agent import OrderAgent
        return OrderAgent(db_config={
            "host": "localhost",
            "port": 3306,
            "user": "root",
            "password": "",
            "database": "test",
        })

    def test_extract_order_id_alpha(self, agent):
        """测试订单号提取 - 字母数字混合"""
        order_id = agent._extract_order_id("查询订单 ORD123456")
        assert order_id == "ORD123456"

    def test_extract_order_id_numeric(self, agent):
        """测试订单号提取 - 纯数字"""
        order_id = agent._extract_order_id("我的订单号是 12345678901")
        assert order_id == "12345678901"

    def test_extract_order_id_none(self, agent):
        """测试订单号提取 - 无订单号"""
        order_id = agent._extract_order_id("我的订单到哪了？")
        assert order_id is None

    def test_map_status(self, agent):
        """测试状态码映射"""
        assert agent._map_order_status(1) == "待付款"
        assert agent._map_order_status("3") == "已发货"
        assert agent._map_order_status("unknown_code") == "unknown_code"

    def test_run_without_order_id(self, agent):
        """测试无订单号时的提示"""
        result = agent.run("查询订单")
        assert "订单号" in result


# ------------------------------------------------------------------ #
#  Recommend Agent 测试
# ------------------------------------------------------------------ #

class TestRecommendAgent:
    """推荐 Agent 测试"""

    @pytest.fixture
    def agent(self):
        from agents.recommend_agent import RecommendAgent
        mock_neo4j = MagicMock()
        mock_llm = MagicMock()
        return RecommendAgent(neo4j_driver=mock_neo4j, llm=mock_llm)

    def test_parse_budget_range(self, agent):
        """测试预算解析 - 范围"""
        result = agent._parse_budget("100-200")
        assert result == (100.0, 200.0)

    def test_parse_budget_single(self, agent):
        """测试预算解析 - 单值"""
        result = agent._parse_budget("500")
        assert result == (0, 500.0)

    def test_parse_budget_int(self, agent):
        """测试预算解析 - 整数"""
        result = agent._parse_budget(300)
        assert result == (0, 300.0)

    def test_in_range(self, agent):
        """测试价格范围判断"""
        assert agent._in_range(150, 100, 200) is True
        assert agent._in_range(50, 100, 200) is False
        assert agent._in_range(250, 100, 200) is False


# ------------------------------------------------------------------ #
#  Chitchat Agent 测试
# ------------------------------------------------------------------ #

class TestChitchatAgent:
    """闲聊 Agent 测试"""

    @pytest.fixture
    def agent(self):
        from agents.chitchat_agent import ChitchatAgent
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="你好！有什么可以帮你的？")
        return ChitchatAgent(llm=mock_llm)

    def test_match_greeting_hello(self, agent):
        """测试问候匹配 - 你好"""
        result = agent._match_greeting("你好")
        assert "你好" in result
        assert "电商" in result

    def test_match_greeting_thanks(self, agent):
        """测试问候匹配 - 谢谢"""
        result = agent._match_greeting("谢谢")
        assert "不客气" in result

    def test_match_greeting_none(self, agent):
        """测试无匹配"""
        result = agent._match_greeting("搜索蓝牙耳机")
        assert result == ""

    def test_run_greeting(self, agent):
        """测试运行 - 问候"""
        result = agent.run("你好")
        assert len(result) > 0


# ------------------------------------------------------------------ #
#  直接运行入口
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    # 不使用 pytest 时直接运行
    print("=" * 60)
    print("  电商智能体单元测试")
    print("=" * 60)

    tests = [
        ("Router - 订单意图", lambda: RouterAgent(MagicMock()).route("查询订单 ORD123")),
        ("Router - 客服意图", lambda: RouterAgent(MagicMock()).route("怎么退货")),
        ("Router - 搜索意图", lambda: RouterAgent(MagicMock()).route("搜索耳机")),
        ("Router - 闲聊意图", lambda: RouterAgent(MagicMock()).route("你好")),
    ]

    passed = 0
    failed = 0
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"  ✅ {name}")
                passed += 1
            else:
                print(f"  ❌ {name} - 返回空")
                failed += 1
        except Exception as e:
            print(f"  ❌ {name} - {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  通过: {passed} | 失败: {failed}")
    print(f"{'=' * 60}")
