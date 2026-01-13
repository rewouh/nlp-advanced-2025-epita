import pydantic
import enum
import typing
from pydantic import BaseModel
from uuid import UUID
from enum import Enum
from typing import List, Union, Optional
from pathlib import Path

class UnitTestType(str, Enum):
    ITEM_GIVEN = "item_given"
    INFO_GIVEN = "info_given"
    QUEST_GIVEN = "quest_given"

class UnitTestItemGiven(BaseModel):
    kind: UnitTestType = UnitTestType.ITEM_GIVEN
    item_name: str

class UnitTestInfoGiven(BaseModel):
    kind: UnitTestType = UnitTestType.INFO_GIVEN
    info: str

class UnitTestQuestGiven(BaseModel):
    kind: UnitTestType = UnitTestType.QUEST_GIVEN
    quest_description: str

class EvaluationTestConfig(BaseModel):
    path : Optional[Path] = None
    scenario_id : str
    npc_id : str
    location_trigger_sentence : Optional[str] = None
    npc_trigger_sentence : Optional[str] = None
    unit_tests : List[Union[
        UnitTestItemGiven,
        UnitTestInfoGiven,
        UnitTestQuestGiven
    ]]
    texts : List[str]

class ResponseMetrics(BaseModel):
    response_time_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
class TestMetrics(BaseModel):
    total_response_time_ms: float = 0.0
    avg_response_time_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    num_responses: int = 0
    responses: List[ResponseMetrics] = []
