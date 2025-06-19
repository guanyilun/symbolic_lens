import re
import logging
from dataclasses import dataclass, field
from typing import Callable
from itertools import combinations, permutations

from sympy import evaluate, Expr, Symbol, sympify, Add, Mul, simplify, true as sympy_true
from sympy.unify import unify

logger = logging.getLogger(__name__)


def build_eval_scope(match_dict, wild_to_slot_map):
    eval_scope = {}
    for wild, slot in wild_to_slot_map.items():
        if slot in match_dict:
            val = match_dict[slot]
            eval_scope[wild] = int(val) if val.is_Integer else float(val) if val.is_Float else val
    return eval_scope


@dataclass(frozen=True)
class Rule:
    """A callable transformation rule with structured, composable conditions."""
    name: str
    lhs: Expr
    rhs: Expr
    slots: list[Symbol]
    wild_to_slot_map: dict[str, Symbol]
    ac_ops: set[type] = field(default_factory=lambda: {Add, Mul})
    condition: Expr | None = None

    @classmethod
    def parse(cls, name: str, rule_str: str, where: str | None = None, symbols_map={}):
        try:
            lhs_str, rhs_str = rule_str.split("->")
        except ValueError:
            raise ValueError("Rule must contain '->' separator.")

        q = re.compile(r"\?([a-zA-Z0-9_]+)")
        wild_names_in_lhs = re.findall(q, lhs_str)
        wild_names_in_where = re.findall(q, where) if where else []
        wild_names = list(dict.fromkeys(wild_names_in_lhs + wild_names_in_where))

        rhs_wild_names = re.findall(q, rhs_str)
        if not set(rhs_wild_names).issubset(set(wild_names_in_lhs)):
            raise ValueError("RHS cannot have wildcards not present in LHS")

        slot_symbols = [Symbol(f"slot_{s}") for s in wild_names]
        wild_to_slot_map = dict(zip(wild_names, slot_symbols))
        slot_map_for_sub = {s: f"slot_{s}" for s in wild_names}

        lhs_eval_str = q.sub(lambda m: slot_map_for_sub[m.group(1)], lhs_str)
        rhs_eval_str = q.sub(lambda m: slot_map_for_sub[m.group(1)], rhs_str)

        condition = None
        if where:
            where_eval_str = q.sub(lambda m: slot_map_for_sub.get(m.group(1), m.group(0)), where)
            condition = sympify(where_eval_str, locals=symbols_map, evaluate=False)

        return cls(
            name=name,
            lhs=sympify(lhs_eval_str, locals=symbols_map, evaluate=False),
            rhs=sympify(rhs_eval_str, locals=symbols_map, evaluate=False),
            condition=condition, 
            slots=slot_symbols,
            wild_to_slot_map=wild_to_slot_map,
        )

    def _check_condition(self, match_dict):
        if self.condition is None:
            logger.debug("    - No condition for this rule.")
            return True
        logger.debug(f"    - Checking condition template: {self.condition}")
        logger.debug(f"    - With match dictionary: {match_dict}")
        resolved_condition = self.condition.xreplace(match_dict)
        logger.debug(f"    - Resolved condition: {resolved_condition}")
        with evaluate(True):
            out = resolved_condition.doit()
            if type(out) is bool: return out
            if out is sympy_true: return True
        return False

    def _ac_match(self, expr, op):
        logger.debug(f"  - Attempting AC match for rule '{self.name}' on '{expr}'")
        expr_args, lhs_args = list(expr.args), list(self.lhs.args)
        if len(expr_args) < len(lhs_args): return None

        for combo_indices in combinations(range(len(expr_args)), len(lhs_args)):
            sub_expr_args = [expr_args[i] for i in combo_indices]
            for p in permutations(sub_expr_args):
                sub_expr = op(*p, evaluate=False)
                matches = list(unify(sub_expr, self.lhs, {}, variables=self.slots))
                if matches:
                    match = matches[0]
                    logger.debug(f"    - Found potential AC sub-match: {match}")
                    if not self._check_condition(match):
                        logger.debug("    - AC sub-match failed condition. Continuing...")
                        continue

                    residue_indices = set(range(len(expr_args))) - set(combo_indices)
                    residue_args = [expr_args[i] for i in residue_indices]
                    transformed_part = self.rhs.xreplace(match)
                    result = op(transformed_part, *residue_args, evaluate=False) if residue_args else transformed_part
                    logger.debug(f"  -> Rule '{self.name}' SUCCESS (AC). Result: {result}")
                    return result
        return None

    def __call__(self, expr):
        logger.debug(f"  - Attempting rule '{self.name}' on '{expr}'")
        op = expr.func
        if op in self.ac_ops and op == self.lhs.func:
            return self._ac_match(expr, op)

        matches = list(unify(expr, self.lhs, {}, variables=self.slots))
        if not matches:
            return None

        match = matches[0]
        logger.debug(f"    - Found potential match: {match}")

        if not self._check_condition(match):
            logger.debug("    - Match failed condition.")
            return None

        with evaluate(False):
            result = self.rhs.xreplace(match)
            logger.debug(f"  -> Rule '{self.name}' SUCCESS. Result: {result}")
            return result


@dataclass(frozen=True)
class RuleSet:
    """A callable that tries a list of rules in order."""
    rules: list[Rule]
    def __call__(self, expr):
        logger.debug(f"Applying ruleset to '{expr}'")
        for rule in self.rules:
            if (result := rule(expr)) is not None:
                return result # Return on first successful rule application
        logger.debug(f"No rules in set matched '{expr}'")
        return None

    def __add__(self, other: 'RuleSet') -> 'RuleSet':
        return RuleSet(self.rules + other.rules)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return RuleSet(self.rules[idx])
        elif isinstance(idx, int):
            return self.rules[idx]
        else:
            raise TypeError(f"Index must be an integer or slice, not {type(idx).__name__}")


@dataclass(frozen=True)
class Postwalk:
    ruleset: RuleSet
    def __call__(self, expr):
        logger.debug(f"Postwalk: Visiting '{expr}'")
        with evaluate(False):
            if not hasattr(expr, 'args') or not expr.args:
                logger.debug("  - Leaf node. Applying ruleset.")
                return self.ruleset(expr) or expr

            logger.debug(f"  - Recursing on args: {expr.args}")
            new_args = [self(arg) for arg in expr.args]

            # Check if we are reconstructing an associative operation
            if expr.func in {Add, Mul}:
                flat_args = []
                for arg in new_args:
                    if arg.func == expr.func:
                        flat_args.extend(arg.args)
                    else:
                        flat_args.append(arg)
                new_args = flat_args

            new_expr = expr.func(*new_args)
            logger.debug(f"  - Applying ruleset to reconstructed expr: {new_expr}")
            return self.ruleset(new_expr) or new_expr


@dataclass(frozen=True)
class Prewalk:
    ruleset: RuleSet
    def __call__(self, expr):
        logger.debug(f"Walking expression: {expr}")
        with evaluate(False):
            logger.debug(f"  -> Applying ruleset to current expr: {expr}")
            transformed = self.ruleset(expr)
            if transformed is not None:
                logger.debug(f"  -> Transformation applied: {transformed}")
                return transformed
            
            if not hasattr(expr, 'args') or not expr.args:
                logger.debug(f"  -> Leaf node, no transformation needed.")
                return expr
            
            logger.debug(f"  -> No transformation. Recursing on args: {expr.args}")
            new_args = [self(arg) for arg in expr.args]
            new_expr = expr.func(*new_args)
            
            logger.debug(f"  -> Reconstructed expr: {new_expr}")
            return new_expr


@dataclass(frozen=True)
class FixedPoint:
    walker: Postwalk | Prewalk
    max_iterations: int = 100
    simplify: bool = False
    def __call__(self, expr):
        logger.info("\n--- Starting FixedPoint Transformation ---")
        current_expr = expr
        for i in range(self.max_iterations):
            logger.info(f"--- Iteration {i+1} ---")
            next_expr_unsimplified = self.walker(current_expr)

            if self.simplify:
                logger.debug(f"  - Simplifying expression: {next_expr_unsimplified}")
                next_expr_simplified = simplify(next_expr_unsimplified)
            else:
                next_expr_simplified = next_expr_unsimplified
            
            logger.debug(f"  - Walker produced: {next_expr_unsimplified}")
            if self.simplify:
                 logger.debug(f"  - Simplified to:   {next_expr_simplified}")

            if next_expr_simplified == current_expr:
                logger.info("--- Expression stabilized. FixedPoint finished. ---")
                return current_expr
            current_expr = next_expr_simplified
        
        logger.warning(f"FixedPoint reached max iterations ({self.max_iterations}).")
        return current_expr


@dataclass(frozen=True)
class Passthrough:
    ruleset: RuleSet
    def __call__(self, expr):
        transformed_expr = self.ruleset(expr)
        return transformed_expr or expr


@dataclass(frozen=True)
class Chain:
    """A walker that applies a sequence of other walkers in order."""
    walkers: list[Callable]
    def __call__(self, expr):
        current_expr = expr
        logger.info("--- Starting Chain Transformation ---")
        logger.debug(f"-> Initial expression: {current_expr}")
        for i, walker in enumerate(self.walkers):
            walker_name = type(walker).__name__
            logger.info(f"-> Executing Stage {i+1}: {walker_name}")
            current_expr = walker(current_expr)
            logger.debug(f"-> Expression is now: {current_expr}")
        
        logger.info("--- Chain Finished ---")
        return current_expr
