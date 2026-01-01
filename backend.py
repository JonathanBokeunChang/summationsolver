import sympy as sp

# Define the symbol 'n' as a positive integer (for proper series behavior)
n = sp.symbols('n', positive=True, integer=True)

def analyze_series(expr_str):
    """Analyze convergence of the series Σ a_n from n=1 to ∞, given a_n expression as a string."""
    # Parse the input expression string into a Sympy expression
    try:
        a_n = sp.sympify(expr_str, locals={'n': n})
    except Exception as e:
        return f"Invalid expression: {e}"
    
    # 1. n-th Term Test for Divergence
    try:
        term_limit = sp.limit(a_n, n, sp.oo)  # limit as n -> infinity
    except Exception as e:
        term_limit = None
    if term_limit is None:
        # If Sympy couldn't compute the limit, try a different approach or mark as undetermined for now
        term_limit_value = None
    else:
        # If the limit is a sympy NaN (not a number, e.g. oscillatory) or infinity, treat appropriately
        if term_limit is sp.nan:
            return "Divergent (n-th term does not approach a single value)"
        if term_limit is sp.oo or term_limit is -sp.oo:
            return "Divergent (n-th term → ∞, does not approach 0)"
        # If we get a finite value, check if it's zero
        try:
            term_limit_value = float(term_limit)  # numerical value if possible
        except Exception:
            term_limit_value = term_limit  # if symbolic (like 1/2 or 0), use it directly
    # If the limit exists and is not zero, series diverges
    if term_limit_value not in (0, None):
        # e.g., limit = c != 0
        return f"Divergent (n-th term → {term_limit_value}, not 0)"
    
    # At this point, either limit is 0 or not determined. If not 0, we've returned.
    # We proceed with other tests for convergence.
    
    # 2. Check for alternating series form (-1)^n or similar.
    is_alternating = False
    # Detect factor of (-1)^n (or (-1)^(n+k))
    for factor in a_n.args:
        # Check if any factor is (-1)**something involving n
        if isinstance(factor, sp.Pow) and factor.base == -1:
            if factor.exp.has(n):
                is_alternating = True
                break
    # Also detect cos(n*pi) pattern, which equals (-1)^n for integer n
    if not is_alternating and a_n.has(sp.cos):
        # Replace cos(n*pi) with (-1)**n, cos(n*pi + pi) with (-1)**(n+1) etc.
        a_n_alt = a_n.xreplace({sp.cos(n*sp.pi): (-1)**n,
                                 sp.cos(n*sp.pi + sp.pi): (-1)**(n+1)})
        if a_n_alt != a_n:
            a_n = a_n_alt
            # If replacement happened, it's alternating
            is_alternating = True
    
    # If the series is alternating, check absolute convergence first
    if is_alternating:
        # Consider b_n = |a_n| for absolute convergence tests
        a_n_abs = sp.simplify(sp.Abs(a_n))
        
        # 2a. Test absolute convergence (apply Ratio or Integral/P-Series on |a_n|)
        abs_conv = None  # will be set to True/False if determined
        # Ratio test on |a_n|
        try:
            abs_ratio = sp.limit(a_n_abs.subs(n, n+1) / a_n_abs, n, sp.oo)
        except Exception:
            abs_ratio = None
        if abs_ratio is not None:
            if abs_ratio is sp.oo:
                abs_conv = False
            elif abs_ratio.is_real:
                try:
                    abs_ratio_val = float(abs_ratio)
                except Exception:
                    abs_ratio_val = abs_ratio  # could be 1 or symbolic
                if abs_ratio_val < 1:
                    return f"Convergent (Absolutely convergent by Ratio Test; L = {abs_ratio_val:.3f} < 1)"
                elif abs_ratio_val > 1:
                    abs_conv = False
                else:
                    abs_conv = None
        
        # If ratio test inconclusive, try Integral test (for positive terms of |a_n|)
        if abs_conv is None:
            try:
                abs_integral = sp.integrate(a_n_abs, (n, 1, sp.oo))
            except Exception:
                abs_integral = None
            if abs_integral is sp.oo:
                abs_conv = False
            elif abs_integral is not None:
                abs_conv = True  # got a finite value for the integral
            # Alternatively, we could attempt p-series analysis here for |a_n|
        
        if abs_conv:
            # Absolutely convergent series
            return "Convergent (Absolutely convergent; series of |a_n| converges)"
        
        # 2b. If not absolutely convergent, apply Alternating Series Test for conditional convergence
        # Check if the positive part b_n = |a_n| is eventually decreasing and tends to 0
        # We already know limit -> 0 from earlier (term_limit_value == 0 in this branch)
        # For monotonic decrease, we'll do a simple heuristic check
        monotonic = None
        try:
            # Check sign of derivative of b(n) for large n
            b = a_n_abs
            db = sp.diff(b, n)
            db_limit = sp.limit(db, n, sp.oo)
            if db_limit < 0:
                monotonic = True
        except Exception:
            pass
        if monotonic is None:
            # Fallback: evaluate b_n at a couple of large values
            try:
                b1 = float(a_n_abs.subs(n, 1000))
                b2 = float(a_n_abs.subs(n, 1100))
                if b2 < b1:
                    monotonic = True
            except Exception:
                monotonic = None
        if monotonic is None:
            # Assume eventually decreasing if we cannot easily determine (common for well-behaved series)
            monotonic = True
        if monotonic:
            # All conditions met for alternating series test:
            return "Convergent (Conditionally convergent by Alternating Series Test)"
        # If not monotonic, we cannot conclusively use the alternating test; continue to other tests (as a fallback).
    
    # 3. Geometric Series Test
    # Check if ratio a_{n+1}/a_n is a constant (geometric series)
    try:
        ratio_expr = sp.simplify(a_n.subs(n, n+1) / a_n)
    except Exception:
        ratio_expr = None
    if ratio_expr is not None and not ratio_expr.has(n):
        # ratio_expr is constant
        r_val = float(ratio_expr)
        if abs(r_val) < 1:
            return f"Convergent (Geometric series with |r| = {abs(r_val):.3f} < 1)"
        else:
            return f"Divergent (Geometric series with |r| = {abs(r_val):.3f} ≥ 1)"
    
    # 4. Integral Test / p-Series Test for positive term series
    # Determine if a_n is eventually positive (for integral test applicability)
    positive_eventually = None
    try:
        # Evaluate a_n for a large n to gauge sign
        test_val = a_n.subs(n, 1000)
        if test_val.is_real:
            positive_eventually = (test_val > 0)
    except Exception:
        positive_eventually = None
    if positive_eventually is None and not a_n.has(sp.sign) and not is_alternating:
        # If we have no sign changes apparent and not alternating, assume it's nonnegative eventually
        positive_eventually = True
    
    if positive_eventually:
        # Try the Integral Test
        try:
            improper_integral = sp.integrate(a_n, (n, 1, sp.oo))
        except Exception:
            improper_integral = None
        if improper_integral is sp.oo:
            return "Divergent (by Integral Test; improper integral diverges)"
        elif improper_integral is not None:
            return "Convergent (by Integral Test; improper integral converges to a finite value)"
        
        # If integral test was inconclusive (could not integrate), try p-series via polynomial degrees
        if a_n.is_rational_function(n):
            num, den = a_n.as_numer_denom()
            deg_num = sp.degree(num, n)
            deg_den = sp.degree(den, n)
            if deg_num is not None and deg_den is not None:
                if deg_den - deg_num > 1:
                    return f"Convergent (p-series comparison; effective p = {deg_den - deg_num} > 1)"
                elif deg_den - deg_num == 1:
                    return "Divergent (p-series comparison; effective p ≈ 1)"
                # If deg_den - deg_num < 1, the terms do not decay fast enough (divergent by comparison to harmonic or worse),
                # but such cases should have been caught by n-th term test if deg_den - deg_num < 0 (terms approach infinity or non-zero constant).
    
    # 5. Ratio Test (for absolute convergence in general)
    try:
        L = sp.limit(sp.Abs(a_n.subs(n, n+1) / a_n), n, sp.oo)
    except Exception:
        L = None
    if L is not None:
        if L is sp.oo:
            return "Divergent (by Ratio Test; limit L → ∞ > 1)"
        if L.is_real:
            try:
                L_val = float(L)
            except Exception:
                L_val = L  # could be 1 or symbolic
            if L_val < 1:
                return f"Convergent (by Ratio Test; L = {L_val:.3f} < 1)"
            elif L_val > 1:
                return f"Divergent (by Ratio Test; L = {L_val:.3f} > 1)"
        # If L == 1 or could not be determined exactly, test is inconclusive
    
    # 6. Root Test
    try:
        R = sp.limit(sp.Abs(a_n) ** (1/n), n, sp.oo)
    except Exception:
        R = None
    if R is not None:
        if R is sp.oo:
            return "Divergent (by Root Test; limit → ∞ > 1)"
        if R.is_real:
            try:
                R_val = float(R)
            except Exception:
                R_val = R
            if R_val < 1:
                return f"Convergent (by Root Test; limit = {R_val:.3f} < 1)"
            elif R_val > 1:
                return f"Divergent (by Root Test; limit = {R_val:.3f} > 1)"
        # If R == 1 or undetermined, inconclusive
    
    # 7. Other oscillatory cases (Dirichlet's test scenario)
    # If a_n contains sin(n) or cos(n) factor (not purely alternating pattern), and the non-oscillatory part decays to 0
    if (a_n.has(sp.sin) or a_n.has(sp.cos)):
        # Check if form is sin(kn)/f(n) or cos(kn)/f(n) with f(n) -> ∞ (so that a_n -> 0)
        # We'll assume partial sums of sin(kn) or cos(kn) are bounded (true for linear k*n arguments)
        # So if |a_n| -> 0, we can conclude by Dirichlet's test.
        if term_limit_value == 0 or term_limit_value is None:  # terms go to 0
            return "Convergent (by Dirichlet's Test; oscillatory terms with decaying amplitude)"
    
    # 8. If none of the tests above reached a conclusion, report inconclusive
    return "Inconclusive – unable to determine with available tests"
