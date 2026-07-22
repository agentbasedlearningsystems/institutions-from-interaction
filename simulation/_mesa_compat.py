"""Mesa ≤ 2.x → 3.x compatibility shim.

Mesa 3.0 removed ``mesa.time.StagedActivation``. The rest of the codebase was
written against it. This module reimplements the subset of its behavior that
SnetSim actually uses, and nothing more.
"""
import random


class StagedActivation:
    """Minimal reimplementation of the Mesa 2.x class.

    Supports: ``stage_list``, ``shuffle``, ``shuffle_between_stages``, and the
    attributes ``time``, ``agents``, ``add``, ``step`` that ``SnetSim`` relies on.
    """

    def __init__(self, model, stage_list, shuffle=False, shuffle_between_stages=False):
        self.model = model
        self.stage_list = list(stage_list)
        self.shuffle = shuffle
        self.shuffle_between_stages = shuffle_between_stages
        self.agents = []
        self.time = 0
        self._stage_time = 1.0 / len(self.stage_list) if self.stage_list else 1.0

    def add(self, agent):
        self.agents.append(agent)

    def step(self):
        agent_order = list(self.agents)
        if self.shuffle:
            random.shuffle(agent_order)
        for stage in self.stage_list:
            if self.shuffle_between_stages:
                random.shuffle(agent_order)
            for agent in agent_order:
                getattr(agent, stage)()
            self.time += self._stage_time
        self.time = round(self.time)
