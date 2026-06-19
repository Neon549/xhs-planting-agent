"""Streamlit 前端"""

# 模块 9 实现

import streamlit as st

st.set_page_config(
    page_title="小红书种草决策 Agent",
    page_icon="🌿",
    layout="wide",
)

import requests
import json
import os

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── 标题 ──────────────────────────────────────────────────────
st.title("🌿 小红书种草决策 Agent")
st.caption("基于 LangGraph + Milvus + Hybrid Retrieval 的多 Agent 推荐系统")

# ── 侧边栏：用户设置 ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 设置")
    user_id = st.text_input("用户 ID", value="user_001")
    session_id = st.text_input("会话 ID", value="session_001")
    use_reranker = st.checkbox("启用 Reranker（需要 GPU）", value=False)

    st.divider()
    st.header("📊 系统状态")
    try:
        stats = requests.get(f"{API_URL}/stats", timeout=2).json()
        st.metric("已索引 Chunks", stats.get("chunks_indexed", 0))
        st.metric("活跃会话", stats.get("sessions_active", 0))
        st.success("API 连接正常")
    except Exception:
        st.error("API 未启动，请先运行 FastAPI")

# ── 主区域 ────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🔍 输入查询")

    # 快捷示例
    st.caption("快捷示例：")
    examples = [
        "iPhone 16 值得买吗",
        "安卓旗舰手机推荐预算3000",
        "家用哑铃怎么选",
        "瑜伽垫推荐",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, use_container_width=True):
            st.session_state["query_input"] = ex

    user_query = st.text_area(
        "输入你的问题",
        value=st.session_state.get("query_input", ""),
        height=100,
        placeholder="例如：想买一台轻薄笔记本，预算5000以内，主要用来办公",
    )

    submit = st.button("🚀 开始推荐", type="primary", use_container_width=True)

with col2:
    st.header("📋 执行过程")
    process_placeholder = st.empty()

# ── 结果区域 ──────────────────────────────────────────────────
result_area = st.container()

if submit and user_query:
    with st.spinner("Agent 正在分析中..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "user_query": user_query,
                    "user_id": user_id,
                    "session_id": session_id,
                    "use_reranker": use_reranker,
                },
                timeout=30,
            )
            data = response.json()

            # 执行过程展示
            with process_placeholder.container():
                steps = data.get("steps_taken", [])
                step_icons = {
                    "query_planner": "🧠",
                    "retrieval_agent": "🔍",
                    "ad_detector": "🛡️",
                    "recommender": "⭐",
                }
                for step in steps:
                    icon = step_icons.get(step, "▶")
                    st.success(f"{icon} {step} 完成")

                # 统计数字
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("召回 Chunks", data.get("retrieved_count", 0))
                m2.metric("软广过滤", data.get("ad_count", 0))
                m3.metric("干净 Chunks", data.get("clean_count", 0))

                # 解析结果
                parsed = data.get("parsed_query", {})
                if parsed:
                    st.caption(
                        f"品类识别：{parsed.get('category', '未知')} | "
                        f"关键词：{', '.join(parsed.get('keywords', [])[:5])}"
                    )

                # 个性化上下文
                if data.get("user_context"):
                    st.info(f"📋 个性化：{data['user_context']}")

            # 推荐结果
            with result_area:
                st.divider()
                st.header("⭐ 推荐结果")
                st.markdown(data.get("final_recommendation", ""))

                # 来源笔记
                sources = data.get("sources", [])
                if sources:
                    st.subheader("📎 推荐依据")
                    for i, src in enumerate(sources):
                        with st.expander(
                            f"来源 {i+1}：{src.get('text_preview', '')[:40]}..."
                        ):
                            st.write(f"**笔记 ID**：{src.get('note_id')}")
                            st.write(f"**点赞数**：{src.get('liked_count', 0)}")
                            st.write(f"**内容**：{src.get('text_preview', '')}")

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ 无法连接 API，请先启动：`uv run uvicorn src.api.main:app --reload`"
            )
        except Exception as e:
            st.error(f"❌ 出错了：{e}")

elif submit and not user_query:
    st.warning("请输入查询内容")
