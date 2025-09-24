#!/usr/bin/env python3

"""
Calculator CLI: safely evaluates arithmetic expressions without using eval.

Supported features:
- Binary operators: +, -, *, /, //, %, **
- Unary operators: +, -
- Parentheses: ( ... )

Examples:
  python3 calculator.py "2 + 2*3"        -> 8
  python3 calculator.py "-(1+2)**3 + 5"  -> -22

Interactive mode (REPL):
  python3 calculator.py
  calc> 2 + 2
  4
  calc> quit
"""

from __future__ import annotations

import argparse
import ast
import operator
import sys
from typing import Union, Callable, Dict


Number = Union[int, float]


class SafeEvaluator(ast.NodeVisitor):
    """Evaluate an arithmetic AST while allowing only safe nodes.

    This visitor computes the numeric value directly instead of building
    an intermediate representation.
    """

    binary_operators: Dict[type, Callable[[Number, Number], Number]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    unary_operators: Dict[type, Callable[[Number], Number]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def visit_Expression(self, node: ast.Expression) -> Number:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> Number:
        left_value = self.visit(node.left)
        right_value = self.visit(node.right)
        operator_type = type(node.op)
        if operator_type not in self.binary_operators:
            raise ValueError(f"Operator not allowed: {operator_type.__name__}")
        operation = self.binary_operators[operator_type]
        try:
            return operation(left_value, right_value)
        except ZeroDivisionError as error:
            raise ZeroDivisionError("division by zero") from error

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Number:
        operand_value = self.visit(node.operand)
        operator_type = type(node.op)
        if operator_type not in self.unary_operators:
            raise ValueError(f"Unary operator not allowed: {operator_type.__name__}")
        operation = self.unary_operators[operator_type]
        return operation(operand_value)

    def visit_Constant(self, node: ast.Constant) -> Number:
        value = node.value
        # Disallow booleans (bool is a subclass of int in Python)
        if isinstance(value, bool):
            raise ValueError("boolean values are not allowed")
        if isinstance(value, (int, float)):
            return value
        raise ValueError(f"constant not allowed: {value!r}")

    # Python <3.8 compatibility (Num nodes) is not strictly needed here, but harmless
    def visit_Num(self, node: ast.Num) -> Number:  # type: ignore[override]
        return self.visit_Constant(ast.Constant(node.n))

    def generic_visit(self, node: ast.AST) -> Number:
        # Block any other node types such as names, calls, attributes, lists, etc.
        raise ValueError(f"syntax element not allowed: {type(node).__name__}")


def evaluate_expression(expression: str) -> Number:
    """Parse and evaluate an arithmetic expression safely.

    Raises ValueError for unsupported syntax and ZeroDivisionError for division by zero.
    """
    try:
        parsed: ast.AST = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        # Normalize to ValueError for a cleaner CLI message
        raise ValueError(f"syntax error: {error.msg}") from error

    evaluator = SafeEvaluator()
    return evaluator.visit(parsed)  # type: ignore[arg-type]


def run_repl() -> int:
    """Run a simple interactive prompt."""
    print("Type expressions to evaluate them. Type 'quit' or 'exit' to leave.")
    while True:
        try:
            line = input("calc> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if line.lower() in {"quit", "exit"}:
            return 0
        if not line:
            continue

        try:
            result = evaluate_expression(line)
            print(result)
        except ZeroDivisionError as error:
            print(f"Error: {error}")
        except ValueError as error:
            print(f"Error: {error}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Safe calculator CLI")
    parser.add_argument(
        "expression",
        nargs="?",
        help="Expression to evaluate (if omitted, starts interactive mode)",
    )
    args = parser.parse_args(argv)

    if args.expression is None:
        return run_repl()

    try:
        value = evaluate_expression(args.expression)
        print(value)
        return 0
    except ZeroDivisionError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

