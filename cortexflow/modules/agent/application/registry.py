"""Agent registry.

Agents are looked up by the name a workflow definition uses, so adding a new
agent is a registration, not an orchestrator change.
"""

from __future__ import annotations

from cortexflow.modules.agent.application.base import BaseAgent
from cortexflow.shared.errors import NotFoundError
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' is already registered")
        self._agents[agent.name] = agent
        logger.info("agent registered", extra={"agent": agent.name})

    def register_all(self, agents: list[BaseAgent]) -> None:
        for agent in agents:
            self.register(agent)

    def get(self, name: str) -> BaseAgent:
        try:
            return self._agents[name]
        except KeyError as exc:
            raise NotFoundError("Unknown agent", agent=name) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._agents))

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": agent.name, "description": agent.description}
            for agent in self._agents.values()
        ]
