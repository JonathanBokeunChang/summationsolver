import math
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor,
    implicit_multiplication_application,
    function_exponentiation,
)

# Define the symbol 'n' as a positive integer (for proper series behavior)
n = sp.symbols('n', positive=True, integer=True)

PARSER_TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
    function_exponentiation,
)
PARSER_LOCALS = {
    'n': n,
    'arctan': sp.atan,
    'atan': sp.atan,
    'arcsin': sp.asin,
    'asin': sp.asin,
    'arccos': sp.acos,
    'acos': sp.acos,
    'ln': sp.log,
    'log': sp.log,
    'sqrt': sp.sqrt,
    'exp': sp.exp,
    'abs': sp.Abs,
    'Abs': sp.Abs,
    'sin': sp.sin,
    'cos': sp.cos,
    'tan': sp.tan,
    'sec': sp.sec,
    'csc': sp.csc,
    'cot': sp.cot,
    'sinh': sp.sinh,
    'cosh': sp.cosh,
    'tanh': sp.tanh,
    'pi': sp.pi,
    'e': sp.E,
    'E': sp.E,
}

def _parse_series_expression(expr_str):
    expr_str = expr_str.strip()
    if not expr_str:
        raise ValueError("Empty expression")
    return parse_expr(
        expr_str,
        local_dict=PARSER_LOCALS,
        transformations=PARSER_TRANSFORMATIONS,
    )

def _safe_limit(expr):
    try:
        limit_expr = sp.limit(expr, n, sp.oo)
    except Exception:
        return None
    if isinstance(limit_expr, sp.Limit):
        try:
            limit_expr = limit_expr.doit()
        except Exception:
            return None
    if isinstance(limit_expr, sp.Limit):
        return None
    return limit_expr

def _limit_is_zero(expr):
    limit_expr = _safe_limit(expr)
    if limit_expr is None:
        return None
    if limit_expr in (sp.oo, -sp.oo, sp.zoo, sp.nan):
        return False
    if isinstance(limit_expr, sp.AccumBounds):
        return False
    if limit_expr.is_number and limit_expr.is_real:
        return limit_expr.is_zero
    return None

def _finite_nonzero_real_limit(expr):
    limit_expr = _safe_limit(expr)
    if limit_expr is None:
        return None
    if limit_expr in (sp.oo, -sp.oo, sp.zoo, sp.nan):
        return None
    if isinstance(limit_expr, sp.AccumBounds):
        return None
    if limit_expr.is_number and limit_expr.is_real:
        if limit_expr.is_zero:
            return None
        return limit_expr
    try:
        limit_val = float(limit_expr)
    except Exception:
        return None
    if math.isfinite(limit_val) and limit_val != 0.0:
        return limit_expr
    return None

def _compare_to_minus_one(value):
    if value is None:
        return None
    if not (value.is_number and value.is_real):
        return None
    diff = sp.simplify(value + 1)
    if diff.is_zero:
        return 0
    if diff.is_positive:
        return 1
    if diff.is_negative:
        return -1
    return None

def _is_linear_in_n(expr):
    try:
        poly = sp.Poly(expr, n)
    except Exception:
        return False
    if poly is None or poly.total_degree() != 1:
        return False
    coeffs = poly.all_coeffs()
    if len(coeffs) != 2:
        return False
    return coeffs[0] != 0

def _is_eventually_decreasing(expr, start=2):
    interval = sp.Interval.Lopen(start, sp.oo)
    try:
        monotonic = sp.is_monotonic(expr, interval, n)
    except NotImplementedError:
        monotonic = None
    if monotonic is True:
        return True
    try:
        d_expr = sp.diff(expr, n)
        d_val = sp.simplify(d_expr.subs(n, start + 1))
        if d_val.is_real and d_val.is_number:
            if d_val < 0:
                return True
            if d_val > 0:
                return False
    except Exception:
        pass
    try:
        v1 = sp.simplify(expr.subs(n, start + 1))
        v2 = sp.simplify(expr.subs(n, start + 2))
        if v1.is_real and v2.is_real:
            diff = sp.simplify(v1 - v2)
            if diff.is_real and diff.is_number:
                if diff > 0:
                    return True
                if diff < 0:
                    return False
            diff_val = float(diff)
            if diff_val > 0:
                return True
            if diff_val < 0:
                return False
    except Exception:
        pass
    return None

def _split_alternating_factor(expr):
    alt_factor = None
    for pow_expr in expr.atoms(sp.Pow):
        if pow_expr.base == -1 and pow_expr.exp.has(n):
            alt_factor = pow_expr
            break
    if alt_factor is None and expr.has(sp.cos):
        replacements = {
            sp.cos(sp.pi * n): (-1) ** n,
            sp.cos(sp.pi * (n + 1)): (-1) ** (n + 1),
            sp.cos(sp.pi * n + sp.pi): (-1) ** (n + 1),
        }
        expr = expr.xreplace(replacements)
        for pow_expr in expr.atoms(sp.Pow):
            if pow_expr.base == -1 and pow_expr.exp.has(n):
                alt_factor = pow_expr
                break
    if alt_factor is None:
        return None, expr
    return alt_factor, sp.simplify(expr / alt_factor)

def _split_linear_oscillation(expr):
    for factor in expr.as_ordered_factors():
        if factor.func in (sp.sin, sp.cos):
            arg = factor.args[0]
            if _is_linear_in_n(arg):
                return factor, sp.simplify(expr / factor)
    return None, expr

def _extract_power_log_exponents(expr, max_log_depth=3):
    expr = sp.simplify(expr)
    powers = expr.as_powers_dict()
    bases = [n]
    log_base = sp.log(n)
    for _ in range(max_log_depth):
        bases.append(log_base)
        log_base = sp.log(log_base)
    exponents = []
    for base in bases:
        exp = powers.get(base, 0)
        if exp is None:
            exp = 0
        exp = sp.sympify(exp)
        if not (exp.is_number and exp.is_real):
            return None
        exponents.append(sp.simplify(exp))
        expr = sp.simplify(expr / (base ** exp))
    return exponents, expr

def _leading_series_term(expr, order=4):
    """Return a simplified leading term for expr as n->∞ using asymptotic series."""
    try:
        series_expr = sp.series(expr, n, sp.oo, order)
    except Exception:
        return None
    try:
        principal = series_expr.removeO()
    except Exception:
        return None
    terms = principal.as_ordered_terms()
    if not terms:
        return None
    leading = sp.simplify(terms[0])
    return None if leading is sp.nan else leading

def _power_log_test(a_n):
    for candidate in (a_n, _leading_series_term(a_n)):
        extracted = _extract_power_log_exponents(candidate) if candidate is not None else None
        if extracted is None:
            continue
        exponents, other = extracted
        other_limit = _finite_nonzero_real_limit(other)
        if other_limit is None:
            continue
        p_cmp = _compare_to_minus_one(exponents[0])
        if p_cmp is None:
            continue
        if p_cmp < 0:
            return "Convergent (by limit comparison to p-series; absolute)"
        if p_cmp > 0:
            return "Divergent (by limit comparison to p-series)"
        for exponent in exponents[1:]:
            log_cmp = _compare_to_minus_one(exponent)
            if log_cmp is None:
                break
            if log_cmp < 0:
                return "Convergent (by limit comparison to logarithmic p-series; absolute)"
            if log_cmp > 0:
                return "Divergent (by limit comparison to logarithmic p-series)"
        else:
            return "Divergent (by limit comparison to logarithmic p-series)"
    return None

def _ratio_test(a_n):
    try:
        ratio_expr = sp.Abs(a_n.subs(n, n + 1) / a_n)
    except Exception:
        return None
    L = _safe_limit(ratio_expr)
    if L is None:
        return None
    if L in (sp.oo, -sp.oo, sp.zoo):
        return "Divergent (by Ratio Test; limit L -> ∞ > 1)"
    if isinstance(L, sp.AccumBounds) or L is sp.nan:
        return None
    if L.is_number and L.is_real:
        if L < 1:
            return f"Convergent (Absolutely convergent by Ratio Test; L = {float(L):.6g} < 1)"
        if L > 1:
            return f"Divergent (by Ratio Test; L = {float(L):.6g} > 1)"
    return None

def _root_test(a_n):
    try:
        root_expr = sp.Abs(a_n) ** (1 / n)
    except Exception:
        return None
    R = _safe_limit(root_expr)
    if R is None:
        return None
    if R in (sp.oo, -sp.oo, sp.zoo):
        return "Divergent (by Root Test; limit -> ∞ > 1)"
    if isinstance(R, sp.AccumBounds) or R is sp.nan:
        return None
    if R.is_number and R.is_real:
        if R < 1:
            return f"Convergent (Absolutely convergent by Root Test; limit = {float(R):.6g} < 1)"
        if R > 1:
            return f"Divergent (by Root Test; limit = {float(R):.6g} > 1)"
    return None

def _log_order_absolute_test(a_n):
    try:
        log_ratio = sp.log(sp.Abs(a_n)) / sp.log(n)
    except Exception:
        return None
    p = _safe_limit(log_ratio)
    if p is None:
        return None
    if p == -sp.oo:
        return "Convergent (Absolutely convergent; faster than any power)"
    if p in (sp.oo, -sp.oo, sp.zoo):
        return None
    if isinstance(p, sp.AccumBounds) or p is sp.nan:
        return None
    if p.is_number and p.is_real:
        cmp_val = _compare_to_minus_one(p)
        if cmp_val is not None and cmp_val < 0:
            return "Convergent (Absolutely convergent by power comparison)"
    return None

def _limit_power_comparison(a_n):
    abs_expr = sp.simplify(sp.Abs(a_n))
    candidate_powers = [sp.Rational(p, 2) for p in range(1, 13)]  # 0.5 to 6
    for p in candidate_powers:
        limit_val = _finite_nonzero_real_limit(abs_expr * (n ** p))
        if limit_val is None:
            continue
        if p > 1:
            return f"Convergent (by limit comparison to 1/n^{float(p):.3g}; absolute)"
        if p == 1:
            return "Divergent (by limit comparison to harmonic series)"
        if p < 1:
            return "Divergent (terms comparable to 1/n^p with p ≤ 1)"
    return None

def _rational_function_test(a_n):
    if not a_n.is_rational_function(n):
        return None
    num, den = a_n.as_numer_denom()
    deg_num = sp.degree(num, n)
    deg_den = sp.degree(den, n)
    if deg_num is None or deg_den is None:
        return None
    diff = deg_den - deg_num
    if diff > 1:
        return "Convergent (by comparison to p-series; rational function)"
    if diff == 1:
        limit_val = _finite_nonzero_real_limit(a_n * n)
        if limit_val is not None:
            return "Divergent (by limit comparison to harmonic series)"
    return None

def _integral_test_absolute(a_n):
    abs_expr = sp.simplify(sp.Abs(a_n))
    monotone = _is_eventually_decreasing(abs_expr)
    if monotone is not True:
        return None
    try:
        integral = sp.integrate(abs_expr, (n, 1, sp.oo))
    except Exception:
        return None
    if isinstance(integral, sp.Integral):
        return None
    if integral in (sp.oo, -sp.oo, sp.zoo):
        return None
    return "Convergent (Absolutely convergent by Integral Test)"

def _alternating_test(a_n):
    alt_factor, rest = _split_alternating_factor(a_n)
    if alt_factor is None:
        return None
    if rest.has(sp.sin) or rest.has(sp.cos):
        return None
    b_n = sp.simplify(sp.Abs(rest))
    limit_zero = _limit_is_zero(b_n)
    if limit_zero is not True:
        return None
    monotone = _is_eventually_decreasing(b_n)
    if monotone is True:
        return "Convergent (Conditionally convergent by Alternating Series Test)"
    return None

def _dirichlet_test(a_n):
    osc_factor, rest = _split_linear_oscillation(a_n)
    if osc_factor is None:
        return None
    b_n = sp.simplify(sp.Abs(rest))
    limit_zero = _limit_is_zero(b_n)
    if limit_zero is not True:
        return None
    monotone = _is_eventually_decreasing(b_n)
    if monotone is True:
        return "Convergent (by Dirichlet's Test; oscillatory terms with decreasing amplitude)"
    return None

def analyze_series(expr_str):
    """Analyze convergence of the series Σ a_n from n=1 to ∞, given a_n expression as a string."""
    # Parse the input expression string into a Sympy expression
    try:
        a_n = _parse_series_expression(expr_str)
    except Exception as e:
        return f"Invalid expression: {e}"
    
    a_n = sp.simplify(a_n)
    if a_n.is_zero:
        return "Convergent (series is identically 0)"

    # 1. n-th Term Test for Divergence
    term_limit = _safe_limit(a_n)
    if term_limit is not None:
        if term_limit in (sp.oo, -sp.oo, sp.zoo):
            return "Divergent (n-th term -> ∞, does not approach 0)"
        if term_limit is sp.nan or isinstance(term_limit, sp.AccumBounds):
            return "Divergent (n-th term does not approach a single value)"
        if term_limit.is_number and term_limit.is_real:
            if not term_limit.is_zero:
                return f"Divergent (n-th term -> {term_limit}, not 0)"

    alt_factor, _ = _split_alternating_factor(a_n)
    osc_factor, _ = _split_linear_oscillation(a_n)
    has_conditional_pattern = (alt_factor is not None) or (osc_factor is not None)
    pending_divergence = None

    absolute_tests = (
        _ratio_test,
        _root_test,
        _power_log_test,
        _rational_function_test,
        _log_order_absolute_test,
        _limit_power_comparison,
        _integral_test_absolute,
    )

    for test_fn in absolute_tests:
        result = test_fn(a_n)
        if result is None:
            continue
        if result.startswith("Convergent"):
            return result
        if not has_conditional_pattern:
            return result
        if pending_divergence is None:
            pending_divergence = result

    # Alternating/oscillatory conditional tests
    result = _alternating_test(a_n)
    if result is not None:
        return result

    result = _dirichlet_test(a_n)
    if result is not None:
        return result

    if pending_divergence is not None:
        return pending_divergence

    return "Inconclusive – unable to determine with available tests"
