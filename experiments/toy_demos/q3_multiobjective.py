"""Synthetic Q3 comparison of exact epsilon constraints and DEAP NSGA-II.

This module deliberately uses a tiny, artificial three-UAV instance.  Exact
enumeration is an audit oracle; the NSGA-II search never reads that oracle.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import combinations, product
from numbers import Integral, Real
from statistics import fmean
from time import perf_counter
from typing import Any

from deap import base, creator, tools

from experiments.toy_demos.common import ToyRunRecord

_LABELS = ("A", "B", "C", "D")
_WEIGHTS = (4, 3, 5, 2, 6, 4)
_CHOICES = (
    (((0, 1), 2), ((2, 3), 3), ((0, 4), 5), ((1, 5), 1)),
    (((1, 2), 2), ((3, 4), 4), ((0, 5), 3), ((2, 4), 5)),
    (((4, 5), 2), ((0, 3), 4), ((1, 4), 3), ((2, 5), 1)),
)
_FITNESS_CLASS_NAME = "MathModelSchoolContestQ3FitnessV1_20260731"
_INDIVIDUAL_CLASS_NAME = "MathModelSchoolContestQ3IndividualV1_20260731"


def _seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("seed must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError("seed must be nonnegative")
    return normalized


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be positive")
    return normalized


@dataclass(frozen=True, slots=True)
class Portfolio:
    """One legal selection: one synthetic bomb candidate per UAV."""

    genes: tuple[int, int, int]
    benefit: int
    risk: int

    def __post_init__(self) -> None:
        normalized_genes = _normalize_genes(self.genes)
        supplied_values = (self.benefit, self.risk)
        for field_name, value in zip(("benefit", "risk"), supplied_values, strict=True):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field_name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
            if float(value) < 0.0:
                raise ValueError(f"{field_name} must be nonnegative")

        expected_benefit, expected_risk = _objective_values(normalized_genes)
        if (float(self.benefit), float(self.risk)) != (
            float(expected_benefit),
            float(expected_risk),
        ):
            raise ValueError("benefit and risk do not match the selected genes")
        object.__setattr__(self, "genes", normalized_genes)
        object.__setattr__(self, "benefit", expected_benefit)
        object.__setattr__(self, "risk", expected_risk)

    @property
    def code(self) -> str:
        return "".join(_LABELS[gene] for gene in self.genes)

    def dominates(self, other: Portfolio) -> bool:
        return (
            self.benefit >= other.benefit
            and self.risk <= other.risk
            and (self.benefit > other.benefit or self.risk < other.risk)
        )


@dataclass(frozen=True, slots=True)
class EpsilonResult:
    risk_limit: float
    feasible: bool
    portfolio: Portfolio | None
    failure_reason: str | None
    record: ToyRunRecord


@dataclass(frozen=True, slots=True)
class Nsga2Run:
    seed: int
    nondominated: tuple[Portfolio, ...]
    coverage: float
    precision: float
    dominated_count: int
    unique_evaluations: int
    backend: str
    record: ToyRunRecord


@dataclass(frozen=True, slots=True)
class MultiSeedAssessment:
    runs: tuple[Nsga2Run, ...]
    mean_coverage: float
    minimum_coverage: float
    mean_precision: float
    mean_jaccard: float
    total_false_positives: int
    backend: str


def _normalize_genes(genes: object) -> tuple[int, int, int]:
    try:
        raw_genes = tuple(genes)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("genes must be an iterable of integers") from error
    if len(raw_genes) != 3:
        raise ValueError("genes must contain exactly three candidate indices")
    if any(isinstance(gene, bool) or not isinstance(gene, Integral) for gene in raw_genes):
        raise TypeError("genes must contain only non-boolean integers")
    normalized = tuple(int(gene) for gene in raw_genes)
    if any(gene not in range(4) for gene in normalized):
        raise ValueError("candidate indices must be in [0, 3]")
    return normalized  # type: ignore[return-value]


def _objective_values(genes: tuple[int, int, int]) -> tuple[int, int]:
    covered: set[int] = set()
    risk = 0
    for uav, gene in enumerate(genes):
        target_indices, candidate_risk = _CHOICES[uav][gene]
        covered.update(target_indices)
        risk += candidate_risk
    benefit = sum(_WEIGHTS[index] for index in covered)
    return benefit, risk


def _evaluate_genes(genes: tuple[int, int, int]) -> Portfolio:
    normalized = _normalize_genes(genes)
    benefit, risk = _objective_values(normalized)
    return Portfolio(normalized, benefit, risk)


def enumerate_portfolios() -> tuple[Portfolio, ...]:
    """Enumerate the 4^3 legal portfolios for the exact audit oracle."""

    return tuple(_evaluate_genes(genes) for genes in product(range(4), repeat=3))


def _nondominated(portfolios: tuple[Portfolio, ...]) -> tuple[Portfolio, ...]:
    front = tuple(
        item
        for item in portfolios
        if not any(other.dominates(item) for other in portfolios if other != item)
    )
    return tuple(sorted(front, key=lambda item: (item.risk, item.benefit, item.code)))


def exact_pareto_front() -> tuple[Portfolio, ...]:
    """Return the exact Pareto front, used only for evaluation and auditing."""

    return _nondominated(enumerate_portfolios())


def solve_epsilon(risk_limit: float, *, seed: int = 0) -> EpsilonResult:
    """Maximize benefit subject to an explicit synthetic risk budget."""

    normalized_seed = _seed(seed)
    if isinstance(risk_limit, bool) or not isinstance(risk_limit, Real):
        raise TypeError("risk_limit must be a real number")
    normalized_limit = float(risk_limit)
    if not math.isfinite(normalized_limit):
        raise ValueError("risk_limit must be finite")
    if normalized_limit < 0.0:
        raise ValueError("risk_limit must be nonnegative")

    started_at = perf_counter()
    feasible = tuple(item for item in enumerate_portfolios() if item.risk <= normalized_limit)
    selected = (
        min(feasible, key=lambda item: (-item.benefit, item.risk, item.code))
        if feasible
        else None
    )
    runtime_s = perf_counter() - started_at
    failure_reason = None if selected is not None else "no portfolio satisfies the risk limit"
    on_exact_front = selected in exact_pareto_front() if selected is not None else False
    record = ToyRunRecord(
        demo_name="q3_multiobjective_epsilon",
        solver="exact enumeration epsilon constraint",
        seed=normalized_seed,
        objective=float(selected.benefit if selected is not None else 0.0),
        runtime_s=runtime_s,
        converged=selected is not None,
        passed_manual_case=on_exact_front,
        failure_reason=failure_reason,
        metadata={
            "risk_limit": normalized_limit,
            "selected_code": selected.code if selected is not None else None,
            "selected_risk": selected.risk if selected is not None else None,
            "enumerated_portfolios": 64,
            "provenance_seed": normalized_seed,
        },
    )
    return EpsilonResult(
        risk_limit=normalized_limit,
        feasible=selected is not None,
        portfolio=selected,
        failure_reason=failure_reason,
        record=record,
    )


def _deap_types() -> tuple[type[Any], type[Any]]:
    expected_weights = (1.0, -1.0)
    if not hasattr(creator, _FITNESS_CLASS_NAME):
        creator.create(_FITNESS_CLASS_NAME, base.Fitness, weights=expected_weights)
    fitness_type = getattr(creator, _FITNESS_CLASS_NAME)
    if (
        not isinstance(fitness_type, type)
        or not issubclass(fitness_type, base.Fitness)
        or tuple(getattr(fitness_type, "weights", ())) != expected_weights
    ):
        raise RuntimeError("incompatible DEAP fitness creator class already registered")

    if not hasattr(creator, _INDIVIDUAL_CLASS_NAME):
        creator.create(_INDIVIDUAL_CLASS_NAME, list, fitness=fitness_type)
    individual_type = getattr(creator, _INDIVIDUAL_CLASS_NAME)
    try:
        compatible_individual = (
            isinstance(individual_type, type)
            and issubclass(individual_type, list)
            and isinstance(individual_type().fitness, fitness_type)
        )
    except (AttributeError, TypeError):
        compatible_individual = False
    if not compatible_individual:
        raise RuntimeError("incompatible DEAP individual creator class already registered")
    return fitness_type, individual_type


def run_nsga2(
    *,
    seed: int,
    population_size: int = 40,
    generations: int = 30,
) -> Nsga2Run:
    """Run a genuine DEAP NSGA-II search without consulting the exact front."""

    normalized_seed = _seed(seed)
    normalized_population = _positive_integer(population_size, "population_size")
    normalized_generations = _positive_integer(generations, "generations")
    if normalized_population < 4 or normalized_population % 4 != 0:
        raise ValueError("population_size must be at least four and divisible by four")

    _, individual_type = _deap_types()
    toolbox = base.Toolbox()
    toolbox.register("candidate_index", random.randint, 0, 3)
    toolbox.register("individual", tools.initRepeat, individual_type, toolbox.candidate_index, 3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    evaluated_genes: set[tuple[int, int, int]] = set()

    def evaluate(individual: list[int]) -> tuple[float, float]:
        genes = tuple(int(gene) for gene in individual)
        portfolio = _evaluate_genes(genes)  # type: ignore[arg-type]
        evaluated_genes.add(portfolio.genes)
        return float(portfolio.benefit), float(portfolio.risk)

    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("tournament", tools.selTournamentDCD)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=3, indpb=0.25)

    prior_random_state = random.getstate()
    started_at = perf_counter()
    try:
        random.seed(normalized_seed)
        population = toolbox.population(n=normalized_population)
        for individual in population:
            individual.fitness.values = toolbox.evaluate(individual)
        population = toolbox.select(population, len(population))

        for _ in range(normalized_generations):
            offspring = tuple(
                toolbox.clone(item)
                for item in toolbox.tournament(population, len(population))
            )
            for first, second in zip(offspring[::2], offspring[1::2], strict=True):
                if random.random() <= 0.9:
                    toolbox.mate(first, second)
                    del first.fitness.values
                    del second.fitness.values
                if random.random() <= 0.3:
                    toolbox.mutate(first)
                    del first.fitness.values
                if random.random() <= 0.3:
                    toolbox.mutate(second)
                    del second.fitness.values
            for individual in offspring:
                if not individual.fitness.valid:
                    individual.fitness.values = toolbox.evaluate(individual)
            population = toolbox.select(population + list(offspring), normalized_population)
    finally:
        random.setstate(prior_random_state)
    runtime_s = perf_counter() - started_at

    unique_population = tuple(
        {_evaluate_genes(tuple(int(gene) for gene in item)) for item in population}  # type: ignore[arg-type]
    )
    approximate_front = _nondominated(unique_population)

    # The exact front is accessed only after the heuristic search has ended.
    exact_front = exact_pareto_front()
    exact_set = set(exact_front)
    approximate_set = set(approximate_front)
    true_positives = len(exact_set & approximate_set)
    coverage = true_positives / len(exact_set)
    precision = true_positives / len(approximate_set) if approximate_set else 0.0
    dominated_count = len(approximate_set - exact_set)
    converged = coverage == 1.0
    failure_reason = (
        None if converged else "heuristic incomplete: not all exact Pareto points found"
    )
    record = ToyRunRecord(
        demo_name="q3_multiobjective_nsga2",
        solver="DEAP NSGA-II",
        seed=normalized_seed,
        objective=max((item.benefit for item in approximate_front), default=0),
        runtime_s=runtime_s,
        converged=converged,
        passed_manual_case=coverage == 1.0 and precision == 1.0,
        failure_reason=failure_reason,
        metadata={
            "coverage": coverage,
            "precision": precision,
            "dominated_count": dominated_count,
            "unique_evaluations": len(evaluated_genes),
            "population_size": normalized_population,
            "generations": normalized_generations,
            "backend": "DEAP NSGA-II",
        },
    )
    return Nsga2Run(
        seed=normalized_seed,
        nondominated=approximate_front,
        coverage=coverage,
        precision=precision,
        dominated_count=dominated_count,
        unique_evaluations=len(evaluated_genes),
        backend="DEAP NSGA-II",
        record=record,
    )


def _jaccard(first: Nsga2Run, second: Nsga2Run) -> float:
    first_codes = {item.code for item in first.nondominated}
    second_codes = {item.code for item in second.nondominated}
    union = first_codes | second_codes
    return len(first_codes & second_codes) / len(union) if union else 1.0


def assess_nsga2(
    *,
    seeds: tuple[int, ...] = (3, 7, 11, 19),
    population_size: int = 40,
    generations: int = 30,
) -> MultiSeedAssessment:
    """Run multiple independent seeds and report discovery quality and stability."""

    if not isinstance(seeds, tuple):
        raise TypeError("seeds must be a tuple")
    if not seeds:
        raise ValueError("seeds must not be empty")
    normalized_seeds = tuple(_seed(seed) for seed in seeds)
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds must be unique")

    runs = tuple(
        run_nsga2(
            seed=seed,
            population_size=population_size,
            generations=generations,
        )
        for seed in normalized_seeds
    )
    pairwise_jaccard = tuple(_jaccard(first, second) for first, second in combinations(runs, 2))
    return MultiSeedAssessment(
        runs=runs,
        mean_coverage=fmean(item.coverage for item in runs),
        minimum_coverage=min(item.coverage for item in runs),
        mean_precision=fmean(item.precision for item in runs),
        mean_jaccard=fmean(pairwise_jaccard) if pairwise_jaccard else 1.0,
        total_false_positives=sum(item.dominated_count for item in runs),
        backend="DEAP NSGA-II",
    )
