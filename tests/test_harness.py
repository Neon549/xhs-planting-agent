"""Harness 单元测试"""
"""Harness 单元测试"""
from src.harness.budget import BudgetManager, BudgetAction

def test_budget_proceed():
    """预算充足时应该返回 PROCEED"""
    mgr = BudgetManager(max_tokens=10000)
    state = mgr.create_state()
    assert mgr.check(state) == BudgetAction.PROCEED

def test_budget_downgrade():
    """使用率超过80%应该返回 DOWNGRADE"""
    mgr = BudgetManager(max_tokens=10000, warning_threshold=0.8)
    state = mgr.create_state()
    mgr.consume(state, prompt_tokens=8000, completion_tokens=500)
    assert mgr.check(state) == BudgetAction.DOWNGRADE

def test_budget_stop_by_tokens():
    """Token 耗尽应该返回 STOP"""
    mgr = BudgetManager(max_tokens=1000)
    state = mgr.create_state()
    mgr.consume(state, prompt_tokens=900, completion_tokens=200)
    assert mgr.check(state) == BudgetAction.STOP

def test_budget_stop_by_calls():
    """调用次数超限应该返回 STOP"""
    mgr = BudgetManager(max_tokens=10000, max_llm_calls=3)
    state = mgr.create_state()
    for _ in range(3):
        mgr.consume(state, prompt_tokens=100, completion_tokens=50)
    assert mgr.check(state) == BudgetAction.STOP