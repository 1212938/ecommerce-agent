"""
BERT 商品分类模型 — 本地推理测试脚本

加载训练好的模型，对一批商品名称进行分类预测，验证模型是否正常工作。

使用方法:
    python scripts/test_classify_model.py

可选参数:
    --model_path  模型路径 (默认: models/product_classification/checkpoint/best)
    --text        单条文本测试
"""
import sys
import os
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="BERT 商品分类模型测试")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "models", "best"
        ),
        help="模型路径",
    )
    parser.add_argument("--text", type=str, default=None, help="单条文本测试")
    return parser.parse_args()


def load_model(model_path):
    """加载模型和 tokenizer"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    print(f"[Test] 加载模型: {model_path}")

    # 加载标签
    labels_file = os.path.join(model_path, "labels.txt")
    with open(labels_file, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    print(f"[Test] 类别数: {len(labels)}")
    print(f"[Test] 类别: {', '.join(labels)}")

    # 加载训练配置 (如果存在)
    config_file = os.path.join(model_path, "training_config.json")
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            train_config = json.load(f)
        print(f"[Test] 训练配置: epochs={train_config.get('epochs')}, "
              f"batch_size={train_config.get('batch_size')}, "
              f"lr={train_config.get('learning_rate')}")
        eval_result = train_config.get("eval_result", {})
        print(f"[Test] 验证集: accuracy={eval_result.get('eval_accuracy', 0):.4f}, "
              f"f1={eval_result.get('eval_f1', 0):.4f}")

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[Test] 使用设备: {device}")
    print()

    return tokenizer, model, labels, device


def predict(text, tokenizer, model, labels, device, top_k=3):
    """对单条文本进行分类预测"""
    import torch

    inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(labels)))

    results = []
    for i in range(top_probs[0].shape[0]):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        results.append((labels[idx], prob))

    return results


def run_tests(model_path):
    """运行批量测试"""
    import torch

    tokenizer, model, labels, device = load_model(model_path)

    # 测试用例: (文本, 期望类别)
    test_cases = [
        # 标准格式 (与训练数据类似)
        ("华为手机数码商品302", "手机数码"),
        ("壳牌汽车用品商品696", "汽车用品"),
        ("老凤祥珠宝首饰商品194", "珠宝首饰"),
        ("优衣库服装鞋包商品572", "服装鞋包"),

        # 真实商品名称 (训练数据中没有的)
        ("iPhone 15 Pro Max 256GB 钛金属", "手机数码"),
        ("戴森 V12 Detect Slim 无线吸尘器", "家用电器"),
        ("兰蔻小黑瓶精华肌底液 50ml", "美妆护肤"),
        ("雀巢咖啡美式速溶咖啡粉 100条", "食品生鲜"),
        ("耐克 Air Force 1 男士运动鞋", "服装鞋包"),
        ("海尔冰箱 BCD-470WGHTD1BGZU1", "家用电器"),
        ("好奇铂金装婴儿纸尿裤 XL码", "母婴用品"),
        ("得力文件夹 A4 双夹文件夹", "图书音像"),

        # 边界/歧义用例
        ("小米充电宝 20000mAh", "手机数码"),  # 也可能是家用电器
        ("茅台飞天 53度 500ml", "食品生鲜"),
        ("施华洛世奇水晶项链", "珠宝首饰"),
        ("美的电饭煲 4L 智能预约", "家用电器"),
    ]

    print("=" * 70)
    print(f"{'文本':<35} {'预测类别':<12} {'置信度':<10} {'期望':<12} {'✓/✗'}")
    print("=" * 70)

    correct = 0
    total = len(test_cases)

    for text, expected in test_cases:
        results = predict(text, tokenizer, model, labels, device, top_k=3)
        top_label, top_prob = results[0]
        is_correct = top_label == expected
        if is_correct:
            correct += 1

        mark = "✓" if is_correct else "✗"
        print(f"{text:<35} {top_label:<12} {top_prob:<10.4f} {expected:<12} {mark}")

        if not is_correct or top_prob < 0.9:
            print(f"  Top-3: ", end="")
            for label, prob in results:
                print(f"{label}({prob:.3f}) ", end="")
            print()

    print("=" * 70)
    print(f"准确率: {correct}/{total} = {correct/total*100:.1f}%")
    print()

    # 混淆分析
    if correct < total:
        print("[分析] 错误案例:")
        for text, expected in test_cases:
            results = predict(text, tokenizer, model, labels, device, top_k=3)
            if results[0][0] != expected:
                print(f"  '{text}' → 预测: {results[0][0]} ({results[0][1]:.3f}), "
                      f"期望: {expected}")


def test_single(text, model_path):
    """单条文本测试"""
    tokenizer, model, labels, device = load_model(model_path)
    results = predict(text, tokenizer, model, labels, device, top_k=5)
    print(f"输入: {text}")
    print(f"Top-5 预测:")
    for label, prob in results:
        bar = "█" * int(prob * 30)
        print(f"  {label:<12} {prob:.4f} {bar}")
    print()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"[Error] 模型路径不存在: {args.model_path}")
        sys.exit(1)

    # 检查依赖
    try:
        import torch
        import transformers
    except ImportError as e:
        print(f"[Error] 缺少依赖: {e}")
        print("请先安装: pip install torch transformers")
        sys.exit(1)

    if args.text:
        test_single(args.text, args.model_path)
    else:
        run_tests(args.model_path)
