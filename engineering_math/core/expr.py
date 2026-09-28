"""변수·인자 수·복소수 정책을 주제별로 지정하는 제한된 SymPy 입력."""
import ast
from functools import lru_cache
import re
import numpy as np
import sympy as sp

FUNCTIONS = {
    'sin': (sp.sin,1,1), 'cos': (sp.cos,1,1), 'tan': (sp.tan,1,1),
    'exp': (sp.exp,1,1), 'log': (sp.log,1,2), 'sqrt': (sp.sqrt,1,1),
    'abs': (sp.Abs,1,1), 'Abs': (sp.Abs,1,1), 'sign': (sp.sign,1,1),
    'sinh': (sp.sinh,1,1), 'cosh': (sp.cosh,1,1), 'tanh': (sp.tanh,1,1),
    'Heaviside': (sp.Heaviside,1,2), 'Piecewise': (sp.Piecewise,1,16),
    'besselj': (sp.besselj,2,2), 'bessely': (sp.bessely,2,2),
    'gamma': (sp.gamma,1,1), 'erf': (sp.erf,1,1), 'atan2': (sp.atan2,2,2),
}
NODES = (ast.Expression,ast.BinOp,ast.UnaryOp,ast.Call,ast.Name,ast.Load,
         ast.Constant,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.UAdd,ast.USub,
         ast.Tuple,ast.Compare,ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.Eq,ast.NotEq,
         ast.BitAnd,ast.BitOr,ast.Invert)


class SymbolicComparisons(ast.NodeTransformer):
    """Python의 == 평가 대신 SymPy의 기호 관계식을 생성합니다."""
    names = {ast.Lt: '__lt', ast.LtE: '__le', ast.Gt: '__gt', ast.GtE: '__ge',
             ast.Eq: '__eq', ast.NotEq: '__ne'}

    def visit_Compare(self, node):
        self.generic_visit(node)
        terms = [node.left, *node.comparators]
        comparisons = [ast.Call(func=ast.Name(id=self.names[type(op)], ctx=ast.Load()),
                               args=[terms[index], terms[index+1]], keywords=[])
                       for index, op in enumerate(node.ops)]
        if len(comparisons) == 1:
            return comparisons[0]
        return ast.Call(func=ast.Name(id='__and', ctx=ast.Load()), args=comparisons, keywords=[])


def symbols_for(variables, allow_complex=False):
    return tuple(sp.Symbol(name) if allow_complex else sp.Symbol(name,real=True) for name in variables)


def parse_expression(text, variables=('x',), allow_complex=False):
    return _parse(text,tuple(variables),bool(allow_complex))


@lru_cache(maxsize=64)
def _parse(text, variables, allow_complex):
    if not isinstance(text,str) or not 1 <= len(text.strip()) <= 1000:
        raise ValueError('수식은 1~1000자 문자열로 입력하십시오.')
    if not variables or len(set(variables)) != len(variables) or any(
        not re.fullmatch('[A-Za-z][A-Za-z0-9]*',v) or v in FUNCTIONS or v in ['pi','E','I','True','False'] for v in variables):
        raise ValueError('변수 이름이 유효하지 않습니다.')
    if not re.fullmatch(r'[A-Za-z0-9_+\-*/().,<>=!&|~\s]+',text):
        raise ValueError('지원하지 않는 문자가 있습니다.')
    local={name:item[0] for name,item in FUNCTIONS.items()}
    local.update(zip(variables,symbols_for(variables,allow_complex)))
    local.update(pi=sp.pi,E=sp.E)
    if allow_complex: local['I']=sp.I
    tree=ast.parse(text,mode='eval')
    if len(list(ast.walk(tree)))>250: raise ValueError('수식이 너무 복잡합니다.')
    for node in ast.walk(tree):
        if not isinstance(node,NODES): raise ValueError('지원하지 않는 수식 문법입니다.')
        if isinstance(node,ast.Name) and node.id not in local:
            raise ValueError(f'지원하지 않는 이름: {node.id}')
        if isinstance(node,ast.Constant) and (type(node.value) not in (int,float,bool) or abs(node.value)>1e100):
            raise ValueError('상수가 유효하지 않습니다.')
        if isinstance(node,ast.Call):
            if not isinstance(node.func,ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
                raise ValueError('지원 함수만 호출할 수 있습니다.')
            _,minimum,maximum=FUNCTIONS[node.func.id]
            if not minimum<=len(node.args)<=maximum: raise ValueError('함수의 인자 개수를 확인하십시오.')
            if node.func.id=='Piecewise' and any(not isinstance(arg,ast.Tuple) or len(arg.elts)!=2 for arg in node.args):
                raise ValueError('Piecewise에는 (식, 조건) 쌍을 입력하십시오.')
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Pow):
            if isinstance(node.right,ast.Constant) and abs(node.right.value)>1000:
                raise ValueError('직접 입력하는 지수의 절댓값은 1000 이하여야 합니다.')
    local.update(__lt=sp.Lt, __le=sp.Le, __gt=sp.Gt, __ge=sp.Ge,
                 __eq=sp.Eq, __ne=sp.Ne, __and=sp.And)
    symbolic_tree = ast.fix_missing_locations(SymbolicComparisons().visit(tree))
    expression=sp.sympify(ast.unparse(symbolic_tree),locals=local)
    if not isinstance(expression,sp.Expr) or expression.free_symbols-set(symbols_for(variables,allow_complex)):
        raise ValueError('지정된 변수의 수식을 입력하십시오.')
    if expression.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
        raise ValueError('수식에 무한대 또는 정의되지 않은 값이 있습니다.')
    if not allow_complex and expression.has(sp.I):
        raise ValueError('이 주제는 실수 함수만 지원합니다.')
    return expression


@lru_cache(maxsize=32)
def compiled(expression, variables=('x',), allow_complex=False):
    # 기본 함수는 NumPy만 사용하며 특수함수에 필요한 경우에만 SciPy를 불러온다.
    modules=['scipy','numpy'] if expression.has(sp.besselj,sp.bessely,sp.gamma,sp.erf) else 'numpy'
    return sp.lambdify(symbols_for(variables,allow_complex),expression,modules=modules)


def evaluate(expression, values, variables=('x',), allow_complex=False):
    arrays=np.broadcast_arrays(*[np.asarray(value, dtype=complex if allow_complex else None) for value in values])
    with np.errstate(all='ignore'):
        result=np.asarray(compiled(expression,tuple(variables),allow_complex)(*arrays))
    if not allow_complex and np.iscomplexobj(result):
        raise ValueError('복소수 값이 발생했습니다. 실수 함수를 입력하십시오.')
    result=np.array(np.broadcast_to(result,arrays[0].shape),dtype=complex if allow_complex else float,copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError('표본에 NaN 또는 inf가 있습니다. 함수와 구간을 확인하십시오.')
    return result


def real_constant(text):
    expression=parse_expression(str(text))
    if expression.free_symbols: raise ValueError('변수가 없는 실수 상수를 입력하십시오.')
    value=float(expression)
    if not np.isfinite(value): raise ValueError('유한한 상수를 입력하십시오.')
    return value
