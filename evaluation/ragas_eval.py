"""RAGAs 评估"""
# 模块 8 实现
"""
RAGAs 评估模块
==============
评估生成质量：忠实度（Faithfulness）和答案相关性（Answer Relevancy）。

需要真实 LLM（DashScope/OpenAI），不能用 mock 跑。
等接入真实数据后运行。

运行前需要设置:
    export DASHSCOPE_API_KEY=sk-xxx
    或在 .env 文件里填入
"""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from loguru import logger


def run_ragas_eval(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
) -> dict:
    """
    运行 RAGAs 评估。

    参数:
        questions: 用户问题列表
        answers:   Agent 生成的回答列表
        contexts:  每个问题对应的检索到的 chunks 列表

    返回:
        {"faithfulness": float, "answer_relevancy": float}

    用法:
        results = run_ragas_eval(
            questions=["iPhone 16 推荐吗"],
            answers=["iPhone 16 Pro 性能强但电池差，价格偏高..."],
            contexts=[["标题: iPhone 16 Pro 用了三个月\n正文: A18 Pro 性能强..."]],
        )
    """
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    dataset = Dataset.from_dict(data)

    logger.info(f"RAGAs 评估开始: {len(questions)} 条")
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

    metrics = {
        "faithfulness": round(float(result["faithfulness"]), 4),
        "answer_relevancy": round(float(result["answer_relevancy"]), 4),
    }
    logger.info(f"RAGAs 结果: {metrics}")
    return metrics