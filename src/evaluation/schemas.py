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
    ATTACKED = "attacked"
    STOPPED_CONVERSATION = "stopped_conversation"

class UnitTestItemGiven(BaseModel):
    kind: UnitTestType = UnitTestType.ITEM_GIVEN
    item_name: str

class UnitTestInfoGiven(BaseModel):
    kind: UnitTestType = UnitTestType.INFO_GIVEN
    info: str

class UnitTestAttacked(BaseModel):
    kind: UnitTestType = UnitTestType.ATTACKED

class UnitTestStoppedConversation(BaseModel):
    kind: UnitTestType = UnitTestType.STOPPED_CONVERSATION

class EvaluationTestConfig(BaseModel):
    path : Optional[Path] = None
    scenario_id : str
    npc_id : str
    location_trigger_sentence : Optional[str] = None
    npc_trigger_sentence : Optional[str] = None
    unit_tests : List[Union[
        UnitTestItemGiven,
        UnitTestInfoGiven,
        UnitTestAttacked,
        UnitTestStoppedConversation
    ]]
    texts : List[str]
