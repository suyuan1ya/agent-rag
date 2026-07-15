from types import SimpleNamespace

from src.agent.fsm import AgentContext, AgentState, AgentStateMachine
from src.agent.orchestrator import AgentOrchestrator


def test_fsm_happy_path():
    ctx = AgentContext(query="test")
    fsm = AgentStateMachine()
    fsm.start(ctx)
    assert fsm.transition(ctx) is AgentState.INTENT_ANALYSIS
    ctx.intent = "qa"
    assert fsm.transition(ctx) is AgentState.STRATEGY_SELECTION
    ctx.strategy = ["hybrid_search"]
    assert fsm.transition(ctx) is AgentState.RETRIEVAL
    ctx.current_tool_index = 1
    assert fsm.transition(ctx) is AgentState.EVALUATION
    ctx.evaluation_sufficient = True
    assert fsm.transition(ctx) is AgentState.ANSWER_GENERATION
    assert fsm.transition(ctx) is AgentState.DONE


def test_fsm_refinement_is_bounded():
    ctx = AgentContext(max_refinements=0)
    fsm = AgentStateMachine()
    fsm.start(ctx)
    fsm.force_state(AgentState.EVALUATION, ctx)
    assert fsm.transition(ctx) is AgentState.ANSWER_GENERATION


def test_retrieval_stays_active_until_all_strategy_tools_run():
    ctx = AgentContext(strategy=["hybrid_search", "evaluate_sufficiency"])
    fsm = AgentStateMachine()
    fsm.start(ctx)
    fsm.force_state(AgentState.RETRIEVAL, ctx)

    ctx.current_tool_index = 1
    assert fsm.transition(ctx) is AgentState.RETRIEVAL
    assert not ctx.error_message

    ctx.current_tool_index = 2
    assert fsm.transition(ctx) is AgentState.EVALUATION


def test_orchestrator_initializes_tools_with_public_llm_client():
    settings = SimpleNamespace(
        dashscope_api_key="test-key",
        llm_base_url="https://example.com/v1",
        llm_model="test-model",
    )
    orchestrator = AgentOrchestrator(SimpleNamespace(), settings=settings)

    orchestrator.initialize()

    assert orchestrator._initialized is True
    assert len(orchestrator.tool_registry) == 8
