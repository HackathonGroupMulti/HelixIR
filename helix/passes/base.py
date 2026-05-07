from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from ..tracer.graph import OpGraph


@dataclass
class PassResult:
    pass_name: str
    recommendations: list[dict]
    summary: str

    def to_dict(self) -> dict:
        return {
            "pass_name": self.pass_name,
            "summary": self.summary,
            "recommendations": self.recommendations,
        }


class OptimizationPass(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def run(self, graph: OpGraph) -> PassResult: ...
