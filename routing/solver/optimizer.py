import sys

import cvxpy as cp
import numpy as np

# CVXPY's solving chain passes full kwargs to the SCIP interface, so solver_opts
# contains the key "solver_opts" (invalid for SCIP). Unwrap so only real params are passed.
def _patch_cvxpy_scip_solver_opts():
    try:
        import cvxpy.reductions.solvers.conic_solvers.scip_conif as _scip_mod
        _orig_set_params = _scip_mod.SCIP._set_params

        def _set_params(self, model, verbose, solver_opts, data, dims):
            solver_opts = solver_opts.get("solver_opts", solver_opts) if isinstance(solver_opts, dict) else solver_opts
            return _orig_set_params(self, model, verbose, solver_opts, data, dims)

        _scip_mod.SCIP._set_params = _set_params
    except Exception:
        pass


_patch_cvxpy_scip_solver_opts()

def optimize_model_assignments_train(a, batches, c, g, l, C, G):
    """
    Optimize query-to-model assignments (boolean) and number of model instances.

    Parameters
    ----------
    a : np.ndarray, shape (N, M)
        Performance estimates for N queries and M models.
    c : np.ndarray, shape (M,)
        Cost per query per model.
    g : np.ndarray, shape (M,)
        GPUs per instance for each model.
    l : np.ndarray, shape (M,)
        Query concurrency per instance for each model.
    C : float
        Total budget.
    G : int
        Total number of available GPUs.

    Returns
    -------
    x_opt : np.ndarray, shape (N, M)
        Optimal query assignments (0 or 1).
    I_opt : np.ndarray, shape (M,)
        Optimal number of instances per model.
    """
    N, M = a.shape
    
    # Variables
    x = cp.Variable((N, M), boolean=True)  # now x is strictly 0 or 1
    I = cp.Variable(M, integer=True)
    
    # Constraints
    constraints = []
    
    # Cost constraint
    constraints.append(cp.sum(cp.multiply(x, c)) <= C)
    
    # GPU constraint (weighted by GPU requirements)
    constraints.append(g @ I <= G)
    
    
    # Concurrency per batch and per model
    # Only apply concurrency constraints to models with limited concurrency (open source)
    concurrency_constraint = []
    for j in range(M):
        if l[j] < 1e9:  # Skip closed source models with unlimited concurrency
            for batch in batches:
                concurrency_constraint.append(cp.sum(x[batch, j]) <= l[j] * I[j])
    
    # Add concurrency constraints to main constraints list
    constraints.extend(concurrency_constraint)
    
    # One model per query
    for i in range(N):
        constraints.append(cp.sum(x[i, :]) == 1)
    
    # Bounds for instances
    constraints.append(I >= 0)
    
    # Objective: maximize average performance
    objective = cp.Maximize(cp.sum(cp.multiply(a, x)) / N)
    
    # Problem
    prob = cp.Problem(objective, constraints)
    
    # Solve using a mixed-integer solver
    prob.solve(solver=cp.GLPK_MI, verbose=False, mip_gap=0.01)
    
    # Check if solution was found
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Solver failed with status: {prob.status}")
    
    # Return optimal values
    return x.value, I.value


def optimize_model_assignments_test(a, c, l, I, C, solver="glpk"):
    """
    Optimize query-to-model assignments (boolean) with fixed model instances.
    Uses simple global concurrency constraints per model (not per batch).

    Parameters
    ----------
    a : np.ndarray, shape (N, M)
        Performance estimates for N queries and M models.
    c : np.ndarray, shape (M,)
        Cost per query per model.
    l : np.ndarray, shape (M,)
        Query concurrency per instance for each model.
    I : np.ndarray, shape (M,)
        Number of instances for each model (fixed, given).
    C : float
        Total budget (total cost across all queries).
    solver : str, optional
        "glpk" (default) for GLPK_MI, or "scip" for SCIP via PySCIPOpt (cvxpy uses pyscipopt).

    Returns
    -------
    x_opt : np.ndarray, shape (N, M)
        Optimal query assignments (0 or 1).
    """
    N, M = a.shape
    
    # Variables
    x = cp.Variable((N, M), boolean=True)  # x[i, j] = 1 if query i is assigned to model j
    
    # Constraints
    constraints = []
    
    # Cost constraint: total cost across all queries must not exceed budget
    constraints.append(cp.sum(cp.multiply(x, c)) <= C)
    
    # Concurrency constraint: total queries assigned to each model cannot exceed its capacity
    # Capacity = l[j] * I[j] (concurrency per instance * number of instances)
    # Only apply to models with limited concurrency (open source)
    for j in range(M):
        if l[j] < 1e9:  # Skip closed source models with unlimited concurrency
            constraints.append(cp.sum(x[:, j]) <= l[j] * I[j])
    
    # One model per query: each query must be assigned to exactly one model
    for i in range(N):
        constraints.append(cp.sum(x[i, :]) == 1)
    
    # Objective: maximize average performance
    objective = cp.Maximize(cp.sum(cp.multiply(a, x)) / N)
    
    # Problem
    prob = cp.Problem(objective, constraints)
    
    if solver == "scip":
        prob.solve(solver=cp.SCIP, verbose=False, solver_opts={"limits/gap": 0.01})
    else:
        prob.solve(solver=cp.GLPK_MI, verbose=False, mip_gap=0.01)
    
    # Check if solution was found
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Solver failed with status: {prob.status}")
    
    # Return optimal values
    return x.value


def _solve_lp_relaxation_upper_bound(a, c, l, I, C):
    """
    Solve the LP relaxation (x continuous in [0,1]) to get an upper bound on the MIP.
    Same constraints and objective as the MIP; returns optimal value or None.
    """
    N, M = a.shape
    x = cp.Variable((N, M), nonneg=True)
    constraints = [x <= 1]
    constraints.append(cp.sum(cp.multiply(x, c)) <= C)
    for j in range(M):
        if l[j] < 1e9:
            constraints.append(cp.sum(x[:, j]) <= l[j] * I[j])
    for i in range(N):
        constraints.append(cp.sum(x[i, :]) == 1)
    objective = cp.Maximize(cp.sum(cp.multiply(a, x)) / N)
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.GLPK, verbose=False)
        if prob.status in ("optimal", "optimal_inaccurate") and prob.value is not None:
            return float(prob.value)
    except Exception:
        pass
    return None


def optimize_model_assignments_test_with_timeout(a, c, l, I, C, time_limit_seconds, solver="glpk"):
    """
    Same as optimize_model_assignments_test but with a solver time limit.
    When the limit is hit, returns the best incumbent solution and bound info.

    Parameters
    ----------
    solver : str, optional
        "glpk" (default) for GLPK_MI, or "scip" for SCIP via PySCIPOpt (cvxpy uses pyscipopt).

    Returns
    -------
    x_opt : np.ndarray or None
        Optimal/incumbent assignments.
    timed_out : bool
        True if the solver stopped due to time limit.
    obj_value : float or None
        Objective value of the returned solution (incumbent).
    best_bound : float or None
        Best bound on the objective from the solver (if available).
    mip_gap_used : float or None
        MIP gap at which the returned solution was found (None if no solution).
    """
    N, M = a.shape
    x = cp.Variable((N, M), boolean=True)
    constraints = []
    constraints.append(cp.sum(cp.multiply(x, c)) <= C)
    for j in range(M):
        if l[j] < 1e9:
            constraints.append(cp.sum(x[:, j]) <= l[j] * I[j])
    for i in range(N):
        constraints.append(cp.sum(x[i, :]) == 1)
    objective = cp.Maximize(cp.sum(cp.multiply(a, x)) / N)
    prob = cp.Problem(objective, constraints)

    time_limit_seconds = max(1, time_limit_seconds)
    tm_lim_ms = int(time_limit_seconds * 1000)
    max_mip_gap = 1.0  # 50% cap
    mip_gap = 0.01
    x_val = None
    use_scip = solver == "scip"
    while mip_gap <= max_mip_gap:
        print(f"Solving with mip_gap: {mip_gap}" + (" (SCIP)" if use_scip else ""), file=sys.stderr)
        if use_scip:
            # PySCIPOpt/SCIP: limits/time in seconds, limits/gap = relative MIP gap
            solver_opts = {"limits/time": time_limit_seconds, "limits/gap": mip_gap}
            try:
                prob.solve(solver=cp.SCIP, verbose=False, solver_opts=solver_opts)
            except Exception:
                if mip_gap >= max_mip_gap:
                    return None, True, None, None, None
                mip_gap = min(mip_gap * 5, max_mip_gap)
                continue
        else:
            solver_opts = {"tm_lim": tm_lim_ms, "mip_gap": mip_gap}
            try:
                prob.solve(solver=cp.GLPK_MI, verbose=False, solver_opts=solver_opts)
            except Exception:
                if mip_gap >= max_mip_gap:
                    return None, True, None, None, None
                mip_gap = min(mip_gap * 5, max_mip_gap)
                continue
        # Lower bound = incumbent objective; upper bound = best relaxation bound (maximization)
        lower = float(prob.value) if prob.value is not None else None
        upper = None
        if hasattr(prob, "solver_stats") and prob.solver_stats is not None:
            extra = getattr(prob.solver_stats, "extra_stats", None)
            if isinstance(extra, dict):
                upper = extra.get("best_bound") or extra.get("mip_best_bound")
            elif extra is not None and hasattr(extra, "get"):
                upper = extra.get("best_bound") or extra.get("mip_best_bound")
        print(f"  lower (incumbent)={lower}, upper (best_bound)={upper}", file=sys.stderr)
        x_val = x.value
        if x_val is not None:
            break
        if mip_gap >= max_mip_gap:
            return None, True, None, None, None
        mip_gap = min(mip_gap * 5, max_mip_gap)

    if x_val is None:
        return None, True, None, None, None

    obj_value = float(prob.value) if prob.value is not None else None
    timed_out = prob.status not in ("optimal", "optimal_inaccurate")

    # Upper bound: solve LP relaxation (x continuous in [0,1])
    best_bound = _solve_lp_relaxation_upper_bound(a, c, l, I, C)
    print(f"  upper bound (LP relaxation)={best_bound}", file=sys.stderr)

    return x_val, timed_out, obj_value, best_bound, mip_gap


def optimize_model_assignments_robust_test(a, c, l, I, C):
    """
    Solve max-min assignment:
        max_x  min_i  sum_j a[i,j] x[i,j]
    subject to cost, concurrency, and one-model-per-query constraints.
    """
    N, M = a.shape

    # Decision variables
    x = cp.Variable((N, M), boolean=True)

    # Auxiliary variable for worst-case (minimum) performance
    t = cp.Variable()

    constraints = []

    # Cost constraint
    constraints.append(cp.sum(cp.multiply(x, c)) <= C)

    # Concurrency constraints
    for j in range(M):
        if l[j] < 1e9:
            constraints.append(cp.sum(x[:, j]) <= l[j] * I[j])

    # One model per query
    for i in range(N):
        constraints.append(cp.sum(x[i, :]) == 1)

    # Max-min constraints: each query performance must be >= t
    for i in range(N):
        constraints.append(cp.sum(cp.multiply(a[i, :], x[i, :])) >= t)

    # Objective: maximize worst-case performance
    objective = cp.Maximize(t)

    prob = cp.Problem(objective, constraints)

    prob.solve(solver=cp.GLPK_MI, verbose=False, mip_gap=0.01)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Solver failed with status: {prob.status}")

    return x.value



def optimize_model_assignments_cost_only(a, c, C, solver="glpk"):
    """
    Optimize query-to-model assignments (boolean) and number of model instances.

    Parameters
    ----------
    a : np.ndarray, shape (N, M)
        Performance estimates for N queries and M models.
    c : np.ndarray, shape (M,)
        Cost per query per model.
    C : float
        Total budget.
    solver : str, optional
        "glpk" (default) for GLPK_MI, or "scip" for SCIP via PySCIPOpt (cvxpy uses pyscipopt).

    Returns
    -------
    x_opt : np.ndarray, shape (N, M)
        Optimal query assignments (0 or 1).
    I_opt : np.ndarray, shape (M,)
        Optimal number of instances per model.
    """
    N, M = a.shape
    
    # Variables
    x = cp.Variable((N, M), boolean=True)  # now x is strictly 0 or 1
    
    # Constraints
    constraints = []
    
    # Cost constraint
    constraints.append(cp.sum(cp.multiply(x, c)) <= C)
    
    
    # One model per query
    for i in range(N):
        constraints.append(cp.sum(x[i, :]) == 1)
    
    # Objective: maximize average performance
    objective = cp.Maximize(cp.sum(cp.multiply(a, x)) / N)
    
    # Problem
    prob = cp.Problem(objective, constraints)
    
    # Solve using a mixed-integer solver
    if solver == "scip":
        prob.solve(solver=cp.SCIP, verbose=False, solver_opts={"limits/gap": 0.01})
    else:
        prob.solve(solver=cp.GLPK_MI, verbose=False, mip_gap=0.01)
    
    # Check if solution was found
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Solver failed with status: {prob.status}")
    
    # Return optimal values
    return x.value