"""
商品分类 Agent — 复用 product_classification 训练好的 BERT 模型

输入商品标题/描述 → 输出 15 个商品类别之一
模型: bert-base-chinese + 加权交叉熵 + 混合精度训练

学习参考: product_classification/src/web/service.py
"""

import os
from typing import List

import torch

from agents.tools.base import BaseAgentTool


class ClassifyAgent(BaseAgentTool):
    """
    商品标题自动分类

    复用 product_classification 项目训练好的 BERT 分类模型
    支持 Top-1 和 Top-K 分类结果输出
    """

    name: str = "classify_agent"
    description: str = "商品分类：根据商品标题文本自动分类到15个商品类别之一，返回分类结果和置信度"

    def __init__(self, model_path: str, labels_path: str):
        super().__init__()
        self.model_path = model_path
        self.labels_path = labels_path

        # 延迟加载模型
        self._tokenizer = None
        self._model = None
        self._labels = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    #  懒加载
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self):
        """懒加载模型和标签"""
        if self._model is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # 加载标签
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                self._labels = [line.strip() for line in f if line.strip()]
        else:
            # 默认 15 类标签
            self._labels = self._default_labels()

        # 加载模型
        if os.path.exists(self.model_path):
            print(f"[ClassifyAgent] 加载模型: {self.model_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self._model.to(self._device)
            self._model.eval()
            print(f"[ClassifyAgent] 模型加载完成，共 {len(self._labels)} 个类别")
        else:
            print(f"[ClassifyAgent] 模型路径不存在: {self.model_path}")
            print("[ClassifyAgent] 将使用规则降级方案")
            self._model = None
            self._tokenizer = None

    def _default_labels(self) -> List[str]:
        """默认 15 类商品标签（与训练模型一致）"""
        return [
            "医药保健",
            "图书音像",
            "宠物用品",
            "家居家装",
            "家用电器",
            "手机数码",
            "服装鞋包",
            "母婴用品",
            "汽车用品",
            "珠宝首饰",
            "礼品鲜花",
            "箱包配饰",
            "美妆护肤",
            "运动户外",
            "食品生鲜",
        ]

    # ------------------------------------------------------------------ #
    #  对外接口
    # ------------------------------------------------------------------ #

    def run(self, query: str, **kwargs) -> str:
        """同步执行商品分类"""
        return self.classify_product(query)

    async def arun(self, **kwargs) -> str:
        """异步执行"""
        return self.run(**kwargs)

    # ------------------------------------------------------------------ #
    #  核心分类逻辑
    # ------------------------------------------------------------------ #

    def classify_product(self, title: str) -> str:
        """
        根据商品标题自动分类到15个商品类别之一

        Args:
            title: 商品标题文本

        Returns:
            分类结果字符串，包含类别名和置信度
        """
        self._ensure_loaded()

        # 预处理
        title = self._preprocess(title)

        # 模型推理
        if self._model is not None and self._tokenizer is not None:
            return self._model_predict(title)
        else:
            # 降级：基于关键词规则的分类
            return self._rule_based_classify(title)

    def get_top_k(self, title: str, k: int = 3) -> List[dict]:
        """
        返回 Top-K 分类结果

        Args:
            title: 商品标题
            k: 返回数量

        Returns:
            [{"category": "...", "confidence": "12.34%"}, ...]
        """
        self._ensure_loaded()

        title = self._preprocess(title)

        if self._model is not None and self._tokenizer is not None:
            inputs = self._tokenizer(
                title, max_length=128, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                top_k_result = torch.topk(probs, k, dim=-1)

            return [
                {
                    "category": self._labels[idx],
                    "confidence": f"{conf:.2%}",
                }
                for idx, conf in zip(top_k_result.indices[0], top_k_result.values[0])
            ]
        else:
            # 降级
            return [{"category": self._rule_based_classify(title), "confidence": "N/A"}]

    def _model_predict(self, title: str) -> str:
        """使用 BERT 模型进行推理"""
        inputs = self._tokenizer(
            title,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()

        category = self._labels[pred_idx] if pred_idx < len(self._labels) else "未知"
        confidence = probs[0][pred_idx].item()

        return f"分类结果: {category} (置信度: {confidence:.2%})"

    def _rule_based_classify(self, title: str) -> str:
        """
        规则降级方案：基于关键词匹配进行分类
        当 BERT 模型不可用时使用

        类别名与 labels.txt (BERT 模型训练标签) 保持一致
        """
        keyword_map = {
            "医药保健": ["药", "保健品", "维生素", "保健", "医用", "口罩", "创可贴", "血压"],
            "图书音像": ["书", "图书", "小说", "教材", "杂志", "CD", "DVD", "音像", "电子书"],
            "宠物用品": ["宠物", "狗粮", "猫粮", "猫砂", "狗", "猫", "鱼缸", "鸟笼", "宠物玩具"],
            "家居家装": [
                "床品",
                "枕头",
                "被子",
                "毛巾",
                "窗帘",
                "地毯",
                "灯",
                "装饰",
                "收纳",
                "墙纸",
                "装修",
            ],
            "家用电器": ["冰箱", "洗衣机", "空调", "电视", "微波炉", "电饭煲", "吸尘器", "电磁炉"],
            "手机数码": [
                "手机",
                "平板",
                "相机",
                "耳机",
                "充电器",
                "数据线",
                "蓝牙",
                "音箱",
                "键盘",
                "鼠标",
                "显示器",
                "打印机",
                "电脑",
                "笔记本",
            ],
            "服装鞋包": [
                "衣服",
                "衬衫",
                "裤子",
                "T恤",
                "裙子",
                "内衣",
                "文胸",
                "鞋",
                "靴",
                "包",
                "箱",
                "背包",
                "手提",
                "外套",
                "夹克",
            ],
            "母婴用品": ["婴儿", "纸尿裤", "奶瓶", "奶粉", "婴儿车", "孕", "宝宝", "母婴", "童装"],
            "汽车用品": [
                "汽车",
                "车载",
                "轮胎",
                "车膜",
                "行车记录仪",
                "车充",
                "座垫",
                "机油",
                "壳牌",
            ],
            "珠宝首饰": ["珠宝", "首饰", "项链", "戒指", "耳环", "手镯", "黄金", "钻石", "银饰"],
            "礼品鲜花": ["礼品", "礼物", "鲜花", "花束", "贺卡", "包装", "礼盒"],
            "箱包配饰": ["箱包", "钱包", "皮带", "围巾", "帽子", "手套", "眼镜", "手表", "配饰"],
            "美妆护肤": [
                "面膜",
                "口红",
                "粉底",
                "精华",
                "乳液",
                "化妆",
                "护肤",
                "香水",
                "防晒",
                "洁面",
            ],
            "运动户外": [
                "运动",
                "健身",
                "跑步",
                "瑜伽",
                "户外",
                "露营",
                "帐篷",
                "登山",
                "骑行",
                "球拍",
            ],
            "食品生鲜": [
                "食品",
                "零食",
                "水果",
                "蔬菜",
                "海鲜",
                "牛肉",
                "猪肉",
                "牛奶",
                "饮料",
                "茶叶",
                "坚果",
                "巧克力",
            ],
        }

        title_lower = title.lower()
        for category, keywords in keyword_map.items():
            for kw in keywords:
                if kw.lower() in title_lower:
                    return f"分类结果: {category} (规则匹配，置信度: N/A)"

        return "分类结果: 其他 (规则匹配，置信度: N/A)"

    def _preprocess(self, text: str) -> str:
        """
        文本预处理（与训练时保持一致）

        - 全角转半角
        - 去除多余空白
        - 去除特殊字符
        """
        import unicodedata

        # 全角转半角
        text = unicodedata.normalize("NFKC", text)

        # 去除多余空白
        text = " ".join(text.split())

        # 去除明显的特殊字符（保留中文、英文、数字、基本标点）
        text = "".join(
            c for c in text if c.isalnum() or c in " ()（）【】[]-—_/,，.。;；:：!！?？%％+"
        )

        return text.strip()
