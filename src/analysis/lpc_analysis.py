import numpy as np
import cvxpy as cp
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")


def autocorrelation(signal, order):
    """
    Compute autocorrelation of a signal up to specified order.
    """
    r = np.correlate(signal, signal, mode='full')
    mid = len(r) // 2
    return r[mid:mid + order + 1]


def levinson_durbin(r, order):
    """
    Levinson-Durbin recursion to solve Yule-Walker equations.
    Returns LPC coefficients.
    """
    a = np.zeros(order + 1)
    e = r[0]
    a[0] = 1.0
    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = -acc / e
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] += k * a[i - j]
        a_new[i] = k
        a = a_new
        e *= (1.0 - k * k)
        if e <= 0:
            break
    return -a[1:]


def l2_lpc_predict(segment, order, max_iters=1000, abstol=1e-5):
    """
    L2-norm LPC prediction using multiple solvers with fallback.
    """
    n = len(segment)
    if n <= order:
        return None, None, None

    X = np.zeros((n - order, order))
    y = segment[order:]
    for i in range(n - order):
        X[i, :] = segment[i:i + order][::-1]

    a = cp.Variable(order)
    objective = cp.Minimize(cp.norm2(y - X @ a)**2)
    prob = cp.Problem(objective)

    solvers = [cp.OSQP, cp.ECOS, cp.SCS]
    solved = False
    coeffs = None
    
    for solver in solvers:
        try:
            if solver == cp.OSQP:
                prob.solve(solver=solver, verbose=False, max_iter=max_iters, eps_abs=abstol)
            elif solver == cp.ECOS:
                prob.solve(solver=solver, verbose=False, max_iters=max_iters, abstol=abstol)
            elif solver == cp.SCS:
                prob.solve(solver=solver, verbose=False, max_iters=max_iters, eps=abstol)
            
            if prob.status in ["optimal", "optimal_inaccurate"]:
                coeffs = a.value
                if coeffs is not None:
                    solved = True
                    break
        except:
            continue
    
    if not solved or coeffs is None:
        return None, None, None

    pred = np.zeros_like(segment, dtype=float)
    for i in range(order, n):
        pred[i] = np.dot(coeffs, segment[i-order:i][::-1])
    residuals = segment - pred
    return coeffs, pred, residuals


def l1_lpc_predict(segment, order, max_iters=1000, abstol=1e-5):
    """
    L1-norm LPC prediction using multiple solvers with fallback.
    """
    n = len(segment)
    if n <= order:
        return None, None, None

    X = np.zeros((n - order, order))
    y = segment[order:]
    for i in range(n - order):
        X[i, :] = segment[i:i + order][::-1]

    a = cp.Variable(order)
    objective = cp.Minimize(cp.norm1(y - X @ a))
    prob = cp.Problem(objective)

    solvers = [cp.OSQP, cp.ECOS, cp.SCS]
    solved = False
    coeffs = None
    
    for solver in solvers:
        try:
            if solver == cp.OSQP:
                prob.solve(solver=solver, verbose=False, max_iter=max_iters, eps_abs=abstol)
            elif solver == cp.ECOS:
                prob.solve(solver=solver, verbose=False, max_iters=max_iters, abstol=abstol)
            elif solver == cp.SCS:
                prob.solve(solver=solver, verbose=False, max_iters=max_iters, eps=abstol)
            
            if prob.status in ["optimal", "optimal_inaccurate"]:
                coeffs = a.value
                if coeffs is not None:
                    solved = True
                    break
        except:
            continue
    
    if not solved or coeffs is None:
        return None, None, None

    pred = np.zeros_like(segment, dtype=float)
    for i in range(order, n):
        pred[i] = np.dot(coeffs, segment[i-order:i][::-1])

    return coeffs, pred, segment - pred


def tune_order(segments, orders, lpc_predict_fn, norm='l2'):
    """
    Optimize LPC order.
    Args:
        segments: List of ECG segments.
        orders: List of orders to test (e.g., [4, 6, 8]).
        lpc_predict_fn: Either l2_lpc_predict or l1_lpc_predict.
        norm: 'l1' or 'l2' to define the error metric.
    Returns:
        Best order and results for all orders.
    """
    results = []
    for order in orders:
        norms = []
        for seg in segments:
            if len(seg) <= order:
                continue
            _, _, resid = lpc_predict_fn(seg, order)
            if resid is None:
                continue
            if norm == 'l2':
                norm_val = np.sqrt(np.mean(resid[order:] ** 2))
            else:
                norm_val = np.mean(np.abs(resid[order:]))
            norms.append(norm_val)
        avg_norm = np.mean(norms) if norms else float('inf')
        results.append((order, avg_norm))

    best_order, best_norm = min(results, key=lambda x: x[1])
    print(f"Best order={best_order} (avg {norm}-norm={best_norm:.4f})")
    return best_order, results


def analyze_lpc_for_segments(segments, order_l2, order_l1, label=""):
    """
    Analyze LPC prediction for all segments.
    
    Returns:
        tuple: (l2_results, l1_results, metrics_dict) where metrics_dict contains mean norms
    """
    l2_results, l1_results = [], []
    for i, segment in enumerate(segments):
        if len(segment) > order_l2:
            coeffs_l2, pred_l2, resid_l2 = l2_lpc_predict(segment, order_l2)
            if (coeffs_l2 is None) or (pred_l2 is None) or (resid_l2 is None):
                continue
            l2_results.append({
                'segment': i,
                'coeffs': coeffs_l2,
                'pred': pred_l2,
                'residuals': resid_l2,
                'l2_norm': np.sqrt(np.mean(resid_l2[order_l2:] ** 2)),
                'l1_norm': np.mean(np.abs(resid_l2[order_l2:]))
            })
        if len(segment) > order_l1:
            coeffs_l1, pred_l1, resid_l1 = l1_lpc_predict(segment, order_l1)
            if (coeffs_l1 is None) or (pred_l1 is None) or (resid_l1 is None):
                continue
            l1_results.append({
                'segment': i,
                'coeffs': coeffs_l1,
                'pred': pred_l1,
                'residuals': resid_l1,
                'l2_norm': np.sqrt(np.mean(resid_l1[order_l1:] ** 2)),
                'l1_norm': np.mean(np.abs(resid_l1[order_l1:]))
            })
    
    metrics = {}
    
    if len(l2_results) > 0:
        mean_l2_L2 = np.mean([r['l2_norm'] for r in l2_results])
        mean_l1_L2 = np.mean([r['l1_norm'] for r in l2_results])
        print(f"---- {label} L2-LPC ----")
        print(f"Mean L2 norm={mean_l2_L2:.5f}, mean L1 norm={mean_l1_L2:.5f}")
        metrics['l2_lpc_mean_l2'] = mean_l2_L2
        metrics['l2_lpc_mean_l1'] = mean_l1_L2
    
    if len(l1_results) > 0:
        mean_l2_L1 = np.mean([r['l2_norm'] for r in l1_results])
        mean_l1_L1 = np.mean([r['l1_norm'] for r in l1_results])
        print(f"---- {label} L1-LPC ----")
        print(f"Mean L2 norm={mean_l2_L1:.5f}, mean L1 norm={mean_l1_L1:.5f}")
        metrics['l1_lpc_mean_l2'] = mean_l2_L1
        metrics['l1_lpc_mean_l1'] = mean_l1_L1
    
    print(f"------------------------")
    return l2_results, l1_results, metrics


def extract_mean_norm(res):
    """
    Returns a float mean-norm from either:
    - a dict like {'mean_norm': ...} (or close variants), or nested under 'summary',
    - a list/tuple of per-segment dicts/numbers,
    - a scalar.
    """
    if isinstance(res, dict):
        for k in ('mean_norm', 'mean', 'avg_norm', 'L1_mean', 'L2_mean'):
            if k in res:
                return float(res[k])
        if 'summary' in res and isinstance(res['summary'], dict):
            for k in ('mean_norm', 'mean', 'avg_norm'):
                if k in res['summary']:
                    return float(res['summary'][k])
        raise KeyError("No mean-like key found in result dict.")

    if isinstance(res, (list, tuple)):
        vals = []
        for r in res:
            if isinstance(r, (int, float, np.floating)):
                vals.append(float(r))
            elif isinstance(r, dict):
                for k in ('mean_norm', 'norm', 'l1_norm', 'l2_norm', 'abs_err'):
                    if k in r:
                        vals.append(float(r[k]))
                        break
        if not vals:
            raise ValueError("Could not derive mean from sequence result.")
        return float(np.mean(vals))

    if isinstance(res, (int, float, np.floating)):
        return float(res)

    raise TypeError(f"Unsupported result type: {type(res)}")


def parse_column(col):
   """Extract predictor name and metric type from column name."""
   parts = col.rsplit(' ', 1)
   if len(parts) == 2:
       predictor = parts[0]
       metric = parts[1]
       return predictor, metric
   return None, None