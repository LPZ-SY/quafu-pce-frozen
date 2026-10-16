"""精确有理数的对角泡利多项式；仅处理 I、Z，不把期望值乘积当算符乘积。"""
import ast
from fractions import Fraction
from numbers import Number


class ZPoly:
    def __init__(self, terms=None):
        self.terms = {int(k): Fraction(v) for k, v in (terms or {}).items() if v}

    @staticmethod
    def scalar(value):
        return value if isinstance(value, ZPoly) else ZPoly({0: Fraction(value)})

    @staticmethod
    def z(*qubits):
        mask = 0
        for q in qubits:
            if q < 0:
                raise ValueError('量子比特编号必须非负')
            mask ^= 1 << q
        return ZPoly({mask: 1})

    def __add__(self, other):
        terms = dict(self.terms)
        for mask, coefficient in self.scalar(other).terms.items():
            terms[mask] = terms.get(mask, 0) + coefficient
        return ZPoly(terms)

    __radd__ = __add__

    def __neg__(self):
        return ZPoly({m: -c for m, c in self.terms.items()})

    def __sub__(self, other):
        return self + (-self.scalar(other))

    def __rsub__(self, other):
        return self.scalar(other) - self

    def __mul__(self, other):
        terms = {}
        for m, c in self.terms.items():
            for n, d in self.scalar(other).terms.items():
                terms[m ^ n] = terms.get(m ^ n, 0) + c * d
        return ZPoly(terms)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, Number):
            raise ValueError('只允许除以常数')
        return self * (Fraction(1) / Fraction(other))

    def __pow__(self, exponent):
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError('指数必须是非负整数')
        result = ZPoly.scalar(1)
        for _ in range(exponent):
            result = result * self
        return result

    def evaluate(self, state):
        return sum(c * (-1 if bin(m & state).count('1') % 2 else 1)
                   for m, c in self.terms.items())

    def rows(self):
        return [dict(mask=m, pauli=' '.join('Z'+str(q) for q in range(m.bit_length())
                     if m >> q & 1) or 'I', coefficient=str(c), order=bin(m).count('1'))
                for m, c in sorted(self.terms.items())]


def compile_expression(expression, variables):
    """白名单表达式：名称、常数、列表、sum、加减乘除和整数次幂；不执行输入代码。"""
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            return variables[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [visit(v) for v in node.elts]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'sum' and len(node.args) == 1 and not node.keywords:
            return sum(visit(node.args[0]), ZPoly())
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return a+b
            if isinstance(node.op, ast.Sub): return a-b
            if isinstance(node.op, ast.Mult): return a*b
            if isinstance(node.op, ast.Div): return ZPoly.scalar(a)/b
            if isinstance(node.op, ast.Pow): return ZPoly.scalar(a)**b
        raise ValueError('不支持的表达式')
    return ZPoly.scalar(visit(ast.parse(expression, mode='eval')))


def assignment_operators(mapping):
    return [(1-ZPoly.z(*edge))/2 for edge in mapping]


def constraints(mapping, demand, capacities, lambda_a=1, lambda_c=1):
    k = len(capacities)
    a = assignment_operators(mapping)
    unique = sum(((sum(a[i*k:(i+1)*k])-1)**2 for i in range(len(demand))), ZPoly())
    equal_capacity = sum(((sum(d*a[i*k+v] for i, d in enumerate(demand))-capacities[v])**2
                          for v in range(k)), ZPoly())
    return lambda_a*unique + lambda_c*equal_capacity


def from_diagonal(values):
    """精确沃尔什展开；将任意有理数对角能量表变为 Z 多项式。"""
    out = [Fraction(v) for v in values]
    if not out or len(out) & (len(out)-1):
        raise ValueError('能量表长度必须是二的整数次幂')
    width = 1
    while width < len(out):
        for start in range(0, len(out), 2*width):
            for j in range(start, start+width):
                a, b = out[j], out[j+width]
                out[j], out[j+width] = a+b, a-b
        width *= 2
    return ZPoly({i: v/len(out) for i, v in enumerate(out) if v})
