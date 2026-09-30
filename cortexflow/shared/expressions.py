"""A deliberately tiny, safe expression language.

Three separate things need to evaluate a human-written condition safely: a
workflow guard, a policy rule, and an agent's row filter. All three take the
expression from a person or a model, so none of them may become a
code-execution vector.

This evaluates a whitelisted subset of Python's AST -- comparisons, boolean
logic, membership, literals and attribute paths into a supplied scope. No
calls, no imports, no attribute access on arbitrary objects.

It lives in the shared kernel rather than in the workflow module because it
belongs to none of the three: putting it in any one of them made the other two
depend on that module, which is how the module graph acquired a cycle.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from cortexflow.shared.errors import ValidationError

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BoolOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Attribute,
    ast.Subscript,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.And,
    ast.Or,
    ast.Not,
    ast.USub,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Load,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.IfExp,
)

_PLACEHOLDER = re.compile(r"\$\{\{(?P<expr>.+?)\}\}")


def _reject_dunder(name: str) -> None:
    """Refuse dunder attribute access.

    Lookups resolve against plain data, so ``x.__class__`` would already yield
    None. Rejecting it outright keeps the sandbox boundary obvious and stops
    anyone reasoning about whether a future container type might expose it.
    """
    if name.startswith("__") or name.endswith("__"):
        raise ValidationError("Dunder attribute access is not permitted", attribute=name)


class _SafeEvaluator(ast.NodeVisitor):
    """Walks the parsed expression, resolving names against a scope mapping."""

    def __init__(self, scope: dict[str, Any]) -> None:
        self._scope = scope

    def visit(self, node: ast.AST) -> Any:
        if not isinstance(node, _ALLOWED_NODES):
            raise ValidationError(
                "Expression uses an unsupported construct",
                construct=type(node).__name__,
            )
        return super().visit(node)

    def generic_visit(self, node: ast.AST) -> Any:  # pragma: no cover - guarded above
        raise ValidationError("Expression uses an unsupported construct",
                              construct=type(node).__name__)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id not in self._scope:
            raise ValidationError("Unknown reference in expression", name=node.id)
        return self._scope[node.id]

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        _reject_dunder(node.attr)
        return _lookup(self.visit(node.value), node.attr)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        return _lookup(self.visit(node.value), self.visit(node.slice))

    def visit_List(self, node: ast.List) -> list[Any]:
        return [self.visit(item) for item in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> tuple[Any, ...]:
        return tuple(self.visit(item) for item in node.elts)

    def visit_Set(self, node: ast.Set) -> set[Any]:
        return {self.visit(item) for item in node.elts}

    def visit_Dict(self, node: ast.Dict) -> dict[Any, Any]:
        return {
            self.visit(k): self.visit(v)
            for k, v in zip(node.keys, node.values, strict=True)
            if k is not None
        }

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        return self.visit(node.body) if self.visit(node.test) else self.visit(node.orelse)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        values = (self.visit(v) for v in node.values)
        if isinstance(node.op, ast.And):
            result: Any = True
            for value in values:
                result = value
                if not value:
                    return value
            return result
        result = False
        for value in values:
            result = value
            if value:
                return value
        return result

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return not operand
        return -operand

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left, right = self.visit(node.left), self.visit(node.right)
        match node.op:
            case ast.Add():
                return left + right
            case ast.Sub():
                return left - right
            case ast.Mult():
                return left * right
            case ast.Div():
                return left / right
        raise ValidationError("Unsupported operator")  # pragma: no cover

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = self.visit(comparator)
            if not _compare(op, left, right):
                return False
            left = right
        return True


_ORDERING = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)


def _compare(op: ast.cmpop, left: Any, right: Any) -> bool:
    """Compare two resolved values.

    Ordering against a missing value yields ``False`` rather than raising.
    Missing paths already resolve to ``None`` -- a guard must be evaluable
    before its upstream step has run -- so ``steps.absent.output.count > 0``
    must answer "no", not explode. Raising here would abort the workflow over
    a typo in a workflow expression, and because the error is a TypeError it
    would classify as permanent and dead-letter the message.

    False is also the safe direction: an unknown value does not satisfy a
    threshold, so the case falls to the policy default, which asks a human.
    """
    if isinstance(op, _ORDERING) and (left is None or right is None):
        return False

    match op:
        case ast.Eq():
            return bool(left == right)
        case ast.NotEq():
            return bool(left != right)
        case ast.Lt():
            return bool(left < right)
        case ast.LtE():
            return bool(left <= right)
        case ast.Gt():
            return bool(left > right)
        case ast.GtE():
            return bool(left >= right)
        case ast.In():
            return left in (right or ())
        case ast.NotIn():
            return left not in (right or ())
        case ast.Is():
            return left is right
        case ast.IsNot():
            return left is not right
    raise ValidationError("Unsupported comparison")  # pragma: no cover


def _lookup(container: Any, key: Any) -> Any:
    """Resolve ``a.b`` / ``a['b']`` over plain data, returning None when absent.

    Missing keys resolve to ``None`` rather than raising: a condition such as
    ``steps.decide.output.flagged`` must be evaluable before that step has run.
    """
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    if isinstance(container, (list, tuple)) and isinstance(key, int):
        return container[key] if -len(container) <= key < len(container) else None
    return getattr(container, str(key), None)


def evaluate(expression: str, scope: dict[str, Any]) -> Any:
    """Evaluate ``expression`` against ``scope``."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValidationError("Malformed expression", expression=expression) from exc
    return _SafeEvaluator(scope).visit(tree)


def evaluate_condition(expression: str, scope: dict[str, Any]) -> bool:
    return bool(evaluate(expression, scope))


def render(value: Any, scope: dict[str, Any]) -> Any:
    """Recursively resolve ``${{ ... }}`` placeholders inside a data structure.

    A string that is *exactly* one placeholder yields the referenced value with
    its type intact (numbers stay numbers); placeholders embedded in a larger
    string are interpolated as text.
    """
    if isinstance(value, str):
        whole = _PLACEHOLDER.fullmatch(value.strip())
        if whole:
            return evaluate(whole.group("expr"), scope)
        return _PLACEHOLDER.sub(
            lambda m: str(evaluate(m.group("expr"), scope) or ""), value
        )
    if isinstance(value, dict):
        return {k: render(v, scope) for k, v in value.items()}
    if isinstance(value, list):
        return [render(v, scope) for v in value]
    return value


def validate_expression(expression: str) -> None:
    """Parse-check an expression at definition-load time, not at runtime."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValidationError("Malformed expression", expression=expression) from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValidationError(
                "Expression uses an unsupported construct",
                expression=expression,
                construct=type(node).__name__,
            )
        if isinstance(node, ast.Attribute):
            _reject_dunder(node.attr)
        if isinstance(node, ast.Name) and (
            node.id.startswith("__") or node.id.endswith("__")
        ):
            raise ValidationError(
                "Dunder names are not permitted", expression=expression
            )


def iter_placeholder_expressions(value: Any) -> list[str]:
    """Collect every ``${{ ... }}`` expression in a structure, for validation."""
    found: list[str] = []
    if isinstance(value, str):
        found.extend(m.group("expr") for m in _PLACEHOLDER.finditer(value))
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(iter_placeholder_expressions(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(iter_placeholder_expressions(item))
    return found
