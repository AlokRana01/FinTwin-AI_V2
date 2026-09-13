"""
FinTwin AI — Multi-Agent Package
================================
Defines the core data contracts, state snapshots, schema containers,
authoritative tool registry, and specialist agents for FinTwin AI's multi-agent system.
"""

from agents.state import (
    UserProfileContext,
    CashFlowState,
    BalanceSheetState,
    GoalItemState,
    DerivedIntelligenceState,
    FinancialState,
    ConversationTurn,
    SessionContext,
)

from agents.schemas import (
    AgentStatus,
    ComputationType,
    AuthoritativeProvenance,
    AuthoritativeData,
    AgentResult,
)

from agents.tool_registry import (
    ToolDefinition,
    ToolRegistry,
    DEFAULT_TOOL_REGISTRY,
)

from agents.financial_intelligence import FinancialIntelligenceAgent
from agents.risk_behaviour import RiskBehaviourAgent
from agents.forecast_goal import ForecastGoalAgent
from agents.scenario_simulation import ScenarioSimulationAgent
from agents.financial_coach import FinancialCoachAgent
from agents.orchestrator import (
    OrchestratorAgent,
    AGENT_REGISTRY,
    DEFAULT_ORCHESTRATOR,
)
from agents.llm import (
    LLMProvider,
    GroqProvider,
    GeminiProvider,
    MockLLMProvider,
    LLMResponse,
    LLMService,
    DEFAULT_LLM_SERVICE,
    LLMError,
    LLMConfigurationError,
    LLMAuthenticationError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMProviderError,
    LLMEmptyResponseError,
)

__all__ = [
    "UserProfileContext",
    "CashFlowState",
    "BalanceSheetState",
    "GoalItemState",
    "DerivedIntelligenceState",
    "FinancialState",
    "ConversationTurn",
    "SessionContext",
    "AgentStatus",
    "ComputationType",
    "AuthoritativeProvenance",
    "AuthoritativeData",
    "AgentResult",
    "ToolDefinition",
    "ToolRegistry",
    "DEFAULT_TOOL_REGISTRY",
    "FinancialIntelligenceAgent",
    "RiskBehaviourAgent",
    "ForecastGoalAgent",
    "ScenarioSimulationAgent",
    "FinancialCoachAgent",
    "OrchestratorAgent",
    "AGENT_REGISTRY",
    "DEFAULT_ORCHESTRATOR",
    "LLMProvider",
    "GroqProvider",
    "GeminiProvider",
    "MockLLMProvider",
    "LLMResponse",
    "LLMService",
    "DEFAULT_LLM_SERVICE",
    "LLMError",
    "LLMConfigurationError",
    "LLMAuthenticationError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMProviderError",
    "LLMEmptyResponseError",
]
