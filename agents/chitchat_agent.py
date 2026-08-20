"""
闲聊 Agent — 处理问候、闲聊和其他无法路由到专用 Agent 的对话

使用 LLM 直接回复，保持友好、专业的电商助手人设
"""

from cachetools import TTLCache

from agents.tools.base import BaseAgentTool


class ChitchatAgent(BaseAgentTool):
    """
    闲聊 Agent

    职责：
    - 处理用户问候（你好、早上好等）
    - 处理闲聊/闲谈
    - 处理无法明确分类意图的输入
    - 引导用户使用电商智能助手的功能
    """

    name: str = "chitchat_agent"
    description: str = "闲聊：处理问候、寒暄、闲聊及其他无法分类到专用 Agent 的对话"

    # 系统人设
    SYSTEM_PROMPT = """你是一个友好、专业的电商智能助手。

你的能力包括：
- 🔍 商品搜索（支持分类和价格过滤）
- 📚 知识图谱问答（品牌、属性、分类关系）
- 🏷️ 商品分类（自动识别商品类别）
- 💡 智能推荐（个性化商品推荐）
- 📦 订单查询（订单状态、物流追踪）
- ❓ 售后服务（退换货政策、FAQ）
- 📊 数据分析（销售趋势、商品排行）

要求：
1. 保持友好、简洁的语气
2. 如果用户意图不明确，引导用户使用上述功能
3. 用中文回答
4. 你只负责闲聊和问候。如果用户问到商品、推荐、价格等购物相关问题，引导他们直接描述需求（如"搜索蓝牙耳机"或"推荐户外装备"），不要自行回答商品相关问题。
"""

    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        # LLM 响应缓存：TTL=5min, maxsize=200，避免高频重复问题反复调用 LLM
        self._llm_cache = TTLCache(maxsize=200, ttl=300)

    # ------------------------------------------------------------------ #
    #  对外接口
    # ------------------------------------------------------------------ #

    def run(self, query: str, **kwargs) -> str:
        """同步执行闲聊回复"""
        return self.chat(query)

    async def arun(self, **kwargs) -> str:
        """异步执行"""
        return self.run(**kwargs)

    # ------------------------------------------------------------------ #
    #  核心逻辑
    # ------------------------------------------------------------------ #

    def chat(self, user_input: str) -> str:
        """
        处理用户闲聊/问候

        Args:
            user_input: 用户输入文本

        Returns:
            回复字符串
        """
        # 快速匹配常见问候
        greeting = self._match_greeting(user_input)
        if greeting:
            return greeting

        # 使用 LLM 生成回复
        if self.llm:
            # 检查缓存
            cache_key = user_input.strip().lower()
            if cache_key in self._llm_cache:
                return self._llm_cache[cache_key]

            prompt = f"{self.SYSTEM_PROMPT}\n\n用户: {user_input}\n\n助手:"
            try:
                response = self.llm.invoke(prompt)
                reply = response.content.strip()
                self._llm_cache[cache_key] = reply
                return reply
            except Exception as e:
                print(f"[ChitchatAgent] LLM 调用失败: {e}")

        # 降级回复
        return self._fallback_reply()

    def _fallback_reply(self) -> str:
        """LLM 不可用且无问候匹配时的兜底回复"""
        return "抱歉，我暂时无法回复。您可以尝试搜索商品、查询订单或了解售后政策。"

    def _match_greeting(self, text: str) -> str:
        """匹配常见问候语并返回预设回复"""
        text_lower = text.lower().strip()

        greeting_map = {
            "你好": "你好！我是电商智能助手 🛒。你可以问我商品搜索、推荐、订单查询、售后政策等问题，随时为你服务！",
            "您好": "您好！我是电商智能助手 🛒。有什么可以帮你的吗？可以搜索商品、查订单、问售后政策等。",
            "早上好": "早上好！☀️ 今天想了解什么商品？我可以帮你搜索、推荐或查询订单。",
            "下午好": "下午好！☀️ 有什么购物需求吗？搜索商品、查推荐、问售后都可以找我。",
            "晚上好": "晚上好！🌙 有什么可以帮你的？商品搜索、智能推荐、订单查询随时在线。",
            "哈喽": "哈喽！👋 我是电商智能助手，帮你搜索商品、查订单、问售后政策都可以！",
            "hello": "Hello! 我是电商智能助手 🛒，可以帮你搜索商品、推荐好物、查询订单和售后政策。",
            "hi": "Hi! 👋 有什么购物问题可以帮你？",
            "谢谢": "不客气！😊 如果还有其他问题，随时问我。祝你购物愉快！",
            "感谢": "不客气！😊 很高兴能帮到你。",
            "再见": "再见！👋 下次有购物问题随时找我。祝生活愉快！",
            "拜拜": "拜拜！👋 随时欢迎回来找我。",
        }

        for keyword, reply in greeting_map.items():
            if keyword in text_lower:
                return reply

        return ""
