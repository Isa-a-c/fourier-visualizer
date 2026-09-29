"""주제 모듈과 화면 사이의 입력·결과 계약."""
from dataclasses import dataclass, field
from typing import Any, Protocol
import numpy as np


@dataclass(frozen=True)
class InputSpec:
    key: str
    label: str
    kind: str
    default: Any
    minimum: float|None = None
    maximum: float|None = None
    choices: tuple = ()
    help: str = ''
    visible_in: tuple = ()


@dataclass
class Result:
    topic: str
    params: dict
    metrics: dict
    tables: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    notices: list = field(default_factory=list)


class Topic(Protocol):
    id: str
    title: str
    description: str
    inputs: tuple[InputSpec,...]
    examples: dict
    # 정적 기호 주제는 supports_animation=False를 선언하여 재생을 비활성화합니다.
    def compute(self,params:dict)->Result: ...
    def figures(self,result:Result)->dict: ...
    def animation(self,result:Result)->dict: ...


def validate_params(specs, params):
    if not isinstance(params,dict): raise ValueError('입력 설정은 객체여야 합니다.')
    allowed={spec.key for spec in specs}
    if set(params)-allowed: raise ValueError('지원하지 않는 설정 항목이 있습니다.')
    normalized={}
    for spec in specs:
        value=params.get(spec.key,spec.default)
        if spec.kind=='int':
            if type(value) is not int: raise ValueError(f'{spec.label}: 정수를 입력하십시오.')
        elif spec.kind=='float':
            if type(value) not in (int,float): raise ValueError(f'{spec.label}: 실수를 입력하십시오.')
            value=float(value)
        elif spec.kind in ('text','constant','orders','choice'):
            if not isinstance(value,str): raise ValueError(f'{spec.label}: 문자열을 입력하십시오.')
            value=value.strip()
            if len(value)>1000: raise ValueError('입력이 너무 깁니다.')
        if spec.kind in ('int','float'):
            if not np.isfinite(value) or (spec.minimum is not None and value<spec.minimum) or (spec.maximum is not None and value>spec.maximum):
                raise ValueError(f'{spec.label}: 허용 범위를 확인하십시오.')
        if spec.kind=='choice' and value not in spec.choices: raise ValueError('지원하지 않는 선택값입니다.')
        normalized[spec.key]=value
    return normalized
