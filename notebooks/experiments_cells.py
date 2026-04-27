# %% [markdown]
# # Лабораторная работа №2 — эксперименты (2.2–2.6 и Трек 4)
# 
# Единый ноутбук: теория (2.2, 2.3), ML-практика (2.4–2.6), исследовательский трек 4 (Cautious L-BFGS).
# 
# Графики отображаются в ноутбуке и сохраняются в `report/figures/` в формате **PNG**.

# %%
%matplotlib inline
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIGURES = ROOT / "report" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
np.set_printoptions(precision=4, suppress=True)

from optimization import (
    gradient_descent,
    newton,
    linear_conjugate_gradients,
    nonlinear_conjugate_gradients,
    hessian_free_newton,
    lbfgs,
    cautious_lbfgs,
)
from oracles import (
    BealeOracle,
    QuadraticOracle,
    LogisticL2Oracle,
    PoissonL2Oracle,
    hess_vec_finite_diff,
)

RNG = np.random.default_rng(42)

# %% [markdown]
# ## 0. Датасеты
# 
# По ТЗ базовые эксперименты выполняются на **том же пакете оракулов и датасетов, что и в Лабораторной работе №1**.

# %%
DATASETS = {
    "classification": [
        ROOT / "data" / "phishing.txt",
        ROOT / "data" / "real-sim" / "real-sim",
    ],
    "regression": [
        ROOT / "data" / "cadata.txt",
        ROOT / "data" / "space_ga.txt",
    ],
}

for task, paths in DATASETS.items():
    print(task)
    for path in paths:
        print("  ", path, "OK" if path.exists() else "MISSING")

# %%
def require_file(path: Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Нет файла данных: {p}")
    return p


def load_libsvm_dataset(path):
    X, y = load_svmlight_file(str(path))
    return X.tocsr(), np.asarray(y, dtype=float)


def make_logistic_oracle(X, y_raw, regcoef=None):
    y = np.where(y_raw > 0, 1.0, -1.0)
    if regcoef is None:
        regcoef = 1.0 / X.shape[0]
    return LogisticL2Oracle(X, y, regcoef), y


def make_poisson_oracle(X, y_raw, regcoef=None):
    y = y_raw.astype(float)
    y = y - y.min() + 1e-3
    if regcoef is None:
        regcoef = 1.0 / X.shape[0]
    return PoissonL2Oracle(X, y, regcoef), y


def get_dataset_oracle(path, task):
    X, y = load_libsvm_dataset(path)
    if task == "classification":
        oracle, y = make_logistic_oracle(X, y)
    else:
        oracle, y = make_poisson_oracle(X, y)
    return X, y, oracle


def relative_grad_sq(history):
    g = np.asarray(history["grad_norm"], dtype=float)
    g0 = max(float(g[0]), 1e-30)
    return (g / g0) ** 2


def rel_grad_tol_squared(eps_linear=1e-6):
    """В optimization.py: ||g||^2 <= tol * ||g0||^2.
    Чтобы выполнялось ||g|| <= eps * ||g0||, нужно tol = eps^2."""
    return float(eps_linear * eps_linear)


# Для п. 2.4–2.6: критерий ||g_k|| <= eps * ||g_0|| при eps = 1e-6
EPS_GRAD = 1e-6
TOLERANCE = EPS_GRAD ** 2
MAX_ITER = 3000

METHOD_LABELS = {
    "gd": "GD",
    "ncg": "NCG (PR)",
    "hfn": "HFN",
    "lbfgs": "L-BFGS (L=10)",
    "newton": "Ньютон",
}


def run_method(name, oracle, x0, trace_micro=False):
    ls_wolfe = {"method": "Wolfe"}
    if name == "gd":
        return gradient_descent(
            oracle, x0, tolerance=TOLERANCE, max_iter=MAX_ITER,
            line_search_options=ls_wolfe, trace=True, display=False,
        )
    if name == "newton":
        return newton(
            oracle, x0, tolerance=TOLERANCE, max_iter=200,
            line_search_options=ls_wolfe, trace=True, display=False,
        )
    if name == "ncg":
        return nonlinear_conjugate_gradients(
            oracle, x0, tolerance=TOLERANCE, max_iter=MAX_ITER,
            line_search_options=ls_wolfe, trace=True, display=False,
        )
    if name == "hfn":
        return hessian_free_newton(
            oracle, x0, tolerance=TOLERANCE, max_iter=MAX_ITER,
            line_search_options={**ls_wolfe, "alpha_0": 1.0}, trace=True, display=False,
        )
    if name == "lbfgs":
        return lbfgs(
            oracle, x0, tolerance=TOLERANCE, max_iter=MAX_ITER, memory_size=10,
            line_search_options=ls_wolfe, trace=True, display=False,
        )
    raise ValueError(name)

# %% [markdown]
# ## 1. Проверка `hess_vec` через конечные разности

# %%
np.random.seed(42)

m, n = 12, 5
X = np.random.randn(m, n)
y_clf = np.random.choice([-1.0, 1.0], size=m)
y_reg = np.abs(np.random.randn(m)) + 0.1

log_oracle = LogisticL2Oracle(X, y_clf, regcoef=1.0 / m)
poi_oracle = PoissonL2Oracle(X, y_reg, regcoef=1.0 / m)

x = np.random.randn(n)
v = np.random.randn(n)

for name, oracle in [("Logistic", log_oracle), ("Poisson", poi_oracle)]:
    hv = oracle.hess_vec(x, v)
    hv_fd = hess_vec_finite_diff(oracle.func, x, v)
    err = np.linalg.norm(hv - hv_fd)
    print(f"{name}: ||hess_vec - finite_diff|| = {err:.6e}")

# %% [markdown]
# ## 2. Эксперимент 2.2 — линейный CG против GD на квадратичных задачах

# %%
def build_diagonal_quadratic(n, kappa):
    diag = np.geomspace(1.0, kappa, n)
    A = np.diag(diag)
    b = np.ones(n)
    return A, b


def quadratic_gd(A, b, x0, tol=1e-8, max_iter=20000):
    oracle = QuadraticOracle(A, b)
    return gradient_descent(
        oracle, x0, tolerance=tol, max_iter=max_iter,
        line_search_options={"method": "Wolfe"}, trace=True
    )


def quadratic_cg(A, b, x0, tol=1e-8, max_iter=None):
    return linear_conjugate_gradients(lambda v: A @ v, b, x0, tolerance=tol, max_iter=max_iter, trace=True)


kappas = [1, 10, 1e2, 1e3, 1e4]
dims = [10, 100, 300]

results_cg = {n: [] for n in dims}
results_gd = {n: [] for n in dims}

for n in dims:
    for kappa in kappas:
        A, b = build_diagonal_quadratic(n, kappa)
        x0 = np.zeros(n)
        _, _, hist_cg = quadratic_cg(A, b, x0)
        _, _, hist_gd = quadratic_gd(A, b, x0, max_iter=5000)
        results_cg[n].append(len(hist_cg["residual_norm"]))
        results_gd[n].append(len(hist_gd["grad_norm"]))

fig = plt.figure()
for n in dims:
    plt.plot(kappas, results_cg[n], marker="o", linewidth=2, label=f"CG, n={n}")
    plt.plot(kappas, results_gd[n], marker="s", linewidth=2, linestyle="--", label=f"GD, n={n}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Число обусловленности κ")
plt.ylabel("Число итераций")
plt.title("Эксперимент 2.2: CG vs GD")
plt.legend()
fig.savefig(FIGURES / "exp22_cg_vs_gd.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Эксперимент 2.3 — влияние размера истории в L-BFGS

# %%
memory_sizes = [0, 1, 5, 10, 20, 50]

task = "classification"
path = DATASETS[task][0]

if path.exists():
    X, y, oracle = get_dataset_oracle(path, task)
    x0 = np.zeros(X.shape[1])
    histories = {}

    for L in memory_sizes:
        if L == 0:
            _, msg, hist = gradient_descent(
                oracle, x0, tolerance=1e-6, max_iter=200,
                line_search_options={"method": "Wolfe"}, trace=True
            )
            histories[f"L={L} (GD)"] = hist
        else:
            _, msg, hist = lbfgs(
                oracle, x0, tolerance=1e-6, max_iter=200, memory_size=L,
                line_search_options={"method": "Wolfe"}, trace=True
            )
            histories[f"L={L}"] = hist

    fig = plt.figure()
    for label, hist in histories.items():
        plt.plot(np.arange(len(hist["grad_norm"])), relative_grad_sq(hist), linewidth=2, label=label)
    plt.yscale("log")
    plt.xlabel("Итерация")
    plt.ylabel(r"$||g_k||^2 / ||g_0||^2$")
    plt.title("Эксперимент 2.3: влияние размера истории L")
    plt.legend()
    fig.savefig(FIGURES / "exp23_lbfgs_memory_vs_iter.png", bbox_inches="tight")
    plt.show()

    fig = plt.figure()
    for label, hist in histories.items():
        plt.plot(hist["time"], relative_grad_sq(hist), linewidth=2, label=label)
    plt.yscale("log")
    plt.xlabel("Время, сек")
    plt.ylabel(r"$||g_k||^2 / ||g_0||^2$")
    plt.title("Эксперимент 2.3: влияние L по времени")
    plt.legend()
    fig.savefig(FIGURES / "exp23_lbfgs_memory_vs_time.png", bbox_inches="tight")
    plt.show()

    final_times = []
    for L, (label, hist) in zip(memory_sizes, histories.items()):
        final_times.append(hist["time"][-1])

    fig = plt.figure()
    plt.plot(memory_sizes, final_times, marker="o", linewidth=2)
    plt.xlabel("Размер истории L")
    plt.ylabel("Итоговое время, сек")
    plt.title("Эксперимент 2.3: итоговое время от L")
    fig.savefig(FIGURES / "exp23_lbfgs_final_time_vs_memory.png", bbox_inches="tight")
    plt.show()
else:
    print("Файл не найден:", path)

# %% [markdown]
# ## 4. Эксперимент 2.4 — сравнение методов (phishing, cadata, real-sim)

# %%
paths_24 = {
    "phishing": (require_file(ROOT / "data" / "phishing.txt"), "classification"),
    "cadata": (require_file(ROOT / "data" / "cadata.txt"), "regression"),
    "real-sim": (require_file(ROOT / "data" / "real-sim" / "real-sim"), "classification"),
}

histories_24 = {}
micro_sources = {}

for slug, (path, task) in paths_24.items():
    X, y_raw = load_libsvm_dataset(path)
    n = X.shape[1]
    x0 = np.zeros(n, dtype=float)

    methods = ["gd", "ncg", "hfn", "lbfgs"]
    if n < 1000:
        methods = methods + ["newton"]
    elif slug == "real-sim" and n > 1000:
        print(
            "[2.4] real-sim: Полный метод Ньютона пропущен из-за огромной размерности признаков (n >> 1000)"
        )

    histories_24[slug] = {}
    for m in methods:
        if task == "classification":
            oracle, _ = make_logistic_oracle(X, y_raw)
        else:
            oracle, _ = make_poisson_oracle(X, y_raw)
        xf, msg, hist = run_method(m, oracle, x0)
        histories_24[slug][m] = hist
        print(f"[2.4] {slug} / {m}: {msg}, итераций (записей в истории) = {len(hist['func'])}")

    for m in ("ncg", "hfn", "lbfgs"):
        micro_sources.setdefault(m, []).append(histories_24[slug][m])

colors = plt.cm.tab10(np.linspace(0, 0.9, 10))

for slug in paths_24:
    hists = histories_24[slug]
    fig, ax = plt.subplots()
    for mi, (m, hist) in enumerate(hists.items()):
        it = np.arange(len(hist["func"]))
        fvals = np.clip(np.asarray(hist["func"], dtype=float), 1e-300, None)
        ax.plot(it, fvals, lw=2, label=METHOD_LABELS.get(m, m), color=colors[mi % 10])
    ax.set_yscale("log")
    ax.set_xlabel("Итерация")
    ax.set_ylabel(r"Значение $f(x_k)$")
    ax.set_title(f"Эксп. 2.4 ({slug}): $f$ от итерации")
    ax.legend()
    fig.savefig(FIGURES / f"exp24_func_vs_iter_{slug}.png", bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots()
    for mi, (m, hist) in enumerate(hists.items()):
        t = np.asarray(hist["time"], dtype=float)
        fvals = np.clip(np.asarray(hist["func"], dtype=float), 1e-300, None)
        ax.plot(t, fvals, lw=2, label=METHOD_LABELS.get(m, m), color=colors[mi % 10])
    ax.set_yscale("log")
    ax.set_xlabel("Время, с")
    ax.set_ylabel(r"Значение $f(x_k)$")
    ax.set_title(f"Эксп. 2.4 ({slug}): $f$ от времени")
    ax.legend()
    fig.savefig(FIGURES / f"exp24_func_vs_time_{slug}.png", bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots()
    for mi, (m, hist) in enumerate(hists.items()):
        t = np.asarray(hist["time"], dtype=float)
        rgsq = np.clip(relative_grad_sq(hist), 1e-300, None)
        ax.plot(t, rgsq, lw=2, label=METHOD_LABELS.get(m, m), color=colors[mi % 10])
    ax.set_yscale("log")
    ax.set_xlabel("Время, с")
    ax.set_ylabel(r"$(\Vert\nabla f(x_k)\Vert / \Vert\nabla f(x_0)\Vert)^2$")
    ax.set_title(f"Эксп. 2.4 ({slug}): относит. кв. нормы градиента от времени")
    ax.legend()
    fig.savefig(FIGURES / f"exp24_gradnorm_vs_time_{slug}.png", bbox_inches="tight")
    plt.show()

print("2.4: PNG сохранены в", FIGURES)

# %% [markdown]
# ## 5. Эксперимент 2.5 — микропрофилирование (NCG, HFN, L-BFGS)

# %%
def mean_time_fractions(histories):
    o_acc, l_acc, s_acc = [], [], []
    for h in histories:
        o = np.asarray(h.get("time_oracle", []), dtype=float)
        l_ = np.asarray(h.get("time_linalg", []), dtype=float)
        s = np.asarray(h.get("time_linesearch", []), dtype=float)
        if o.size <= 1:
            continue
        o, l_, s = o[1:], l_[1:], s[1:]
        tot = o + l_ + s
        mask = tot > 0
        if not np.any(mask):
            continue
        o_acc.append(np.mean(o[mask] / tot[mask]))
        l_acc.append(np.mean(l_[mask] / tot[mask]))
        s_acc.append(np.mean(s[mask] / tot[mask]))
    if not o_acc:
        return 0.33, 0.33, 0.34
    return float(np.mean(o_acc)), float(np.mean(l_acc)), float(np.mean(s_acc))


fracs = {}
for m in ("ncg", "hfn", "lbfgs"):
    fracs[m] = mean_time_fractions(micro_sources[m])

methods_order = ("ncg", "hfn", "lbfgs")
labels_ru = [METHOD_LABELS[m] for m in methods_order]
O = np.array([fracs[m][0] for m in methods_order])
L = np.array([fracs[m][1] for m in methods_order])
S = np.array([fracs[m][2] for m in methods_order])

xpos = np.arange(len(methods_order))
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(xpos, O, label="Оракул (f, grad; для HFN + hess_vec)", color="#4C72B0")
ax.bar(xpos, L, bottom=O, label="Лин. алгебра", color="#55A868")
ax.bar(xpos, S, bottom=O + L, label="Линейный поиск", color="#C44E52")
ax.set_xticks(xpos)
ax.set_xticklabels(labels_ru)
ax.set_ylabel("Средняя доля времени за итерацию")
ax.set_title("Эксп. 2.5: микропрофилирование (phishing, cadata, real-sim)")
ax.legend(loc="upper right")
fig.savefig(FIGURES / "exp25_microprofiling.png", bbox_inches="tight")
plt.show()

print("2.5: доли (oracle, linalg, linesearch):", fracs)

# %% [markdown]
# ## 6. Эксперимент 2.6 — оптимизация vs качество на тесте

# %%
def exp26_useless_time_stats(hist, task, metric_tol=1e-4):
    """Доля времени после первого достижения почти оптимальной тестовой метрики."""
    tm = np.asarray(hist["test_metric"], dtype=float)
    t = np.asarray(hist["time"], dtype=float)
    if tm.size == 0 or t.size == 0:
        return 0, 0.0, float(t[-1]) if t.size else 0.0, 0.0
    if task == "classification":
        best = float(np.max(tm))
        idx = np.where(tm >= best - metric_tol)[0]
    else:
        best = float(np.min(tm))
        idx = np.where(tm <= best + metric_tol)[0]
    i_best = int(idx[0]) if len(idx) else int(tm.size - 1)
    t_best = float(t[i_best])
    t_total = float(t[-1])
    useless_pct = 100.0 * (t_total - t_best) / t_total if t_total > 0 else 0.0
    return i_best, t_best, t_total, useless_pct


def exp26_run(path, task, slug, lambda_multipliers=(1.0, 1e-3, 1e-6)):
    X, y_raw = load_libsvm_dataset(path)
    strat = y_raw if task == "classification" else None
    X_tr, X_te, ytr, yte = train_test_split(
        X, y_raw, test_size=0.2, random_state=42, stratify=strat
    )
    m_tr = X_tr.shape[0]
    n = X_tr.shape[1]
    x0 = np.zeros(n, dtype=float)

    if task == "classification":
        y_tr_sign = np.where(ytr > 0, 1.0, -1.0)
        y_te_sign = np.where(yte > 0, 1.0, -1.0)
    else:
        ytr_f = ytr.astype(float)
        yte_f = yte.astype(float)
        base = ytr_f.min()
        y_tr_shift = ytr_f - base + 1e-3
        y_te_shift = yte_f - base + 1e-3

    histories = {}
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(lambda_multipliers)))

    for mult in lambda_multipliers:
        regcoef = float(mult) / m_tr
        if task == "classification":
            oracle = LogisticL2Oracle(X_tr, y_tr_sign, regcoef)

            def callback(x, kw, Xte=X_te, yte_=y_te_sign):
                pred = np.sign(Xte @ x)
                acc = accuracy_score(yte_, pred)
                return {"test_metric": float(acc)}
        else:
            oracle = PoissonL2Oracle(X_tr, y_tr_shift, regcoef)

            def callback(x, kw, Xte=X_te, yte_=y_te_shift):
                z = Xte @ x
                z = np.minimum(z, 50.0)
                pred = np.exp(z)
                mse = mean_squared_error(yte_, pred)
                return {"test_metric": float(mse)}

        xf, msg, hist = lbfgs(
            oracle, x0.copy(), tolerance=TOLERANCE, max_iter=MAX_ITER, memory_size=10,
            line_search_options={"method": "Wolfe"}, trace=True, callback=callback, display=False,
        )
        histories[mult] = hist
        i_b, t_b, t_tot, useless = exp26_useless_time_stats(hist, task, metric_tol=1e-4)
        print(
            f"[2.6] {slug} λ = {mult:g}/m: {msg}, точек = {len(hist['func'])}, "
            f"бесполезное время = {useless:.2f}% (iter_best={i_b}, t_best={t_b:.4f} с, t_total={t_tot:.4f} с)"
        )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for ci, mult in enumerate(lambda_multipliers):
        hist = histories[mult]
        it = np.arange(len(hist["func"]))
        ftr = np.clip(np.asarray(hist["func"], dtype=float), 1e-300, None)
        tm = np.asarray(hist["test_metric"], dtype=float)
        lbl = fr"$\lambda = {mult:g}/m$"
        ax1.plot(it, ftr, lw=2, color=colors[ci], label=lbl)
        ax2.plot(it, tm, lw=2, color=colors[ci], label=lbl)

    ax1.set_yscale("log")
    ax1.set_ylabel("Train loss")
    ax1.legend(loc="best", fontsize=8)
    ax1.set_title(f"Эксп. 2.6 ({slug}): train loss для разных $\\lambda$")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Итерация")
    met = "Accuracy" if task == "classification" else "MSE"
    ax2.set_ylabel(f"Метрика на тесте ({met})")
    ax2.legend(loc="best", fontsize=8)
    ax2.set_title(f"Эксп. 2.6 ({slug}): {met} на тесте для разных $\\lambda$")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / f"exp26_train_vs_quality_{slug}.png", bbox_inches="tight")
    plt.show()

    return {"by_multiplier": histories, "multipliers": list(lambda_multipliers)}


out26_clf = exp26_run(paths_24["phishing"][0], "classification", "clf")
out26_reg = exp26_run(paths_24["cadata"][0], "regression", "reg")
hist26_clf = out26_clf["by_multiplier"][1.0]
hist26_reg = out26_reg["by_multiplier"][1.0]

# %% [markdown]
# ## 7. Исследовательский трек 4 — Cautious L-BFGS vs L-BFGS (Бил + LogisticL2, phishing)

# %%
# Трек 4 (а): функция Била; x0 ~ N(0, 10) (дисперсия 10 по осям)
n_b = 2
x0_beale = RNG.normal(0.0, np.sqrt(10.0), size=n_b)

oracle_w = BealeOracle()
xf_w, msg_w, hist_w = lbfgs(
    oracle_w, x0_beale.copy(), tolerance=1e-8, max_iter=500, memory_size=10,
    line_search_options={"method": "Wolfe"}, trace=True, display=False,
)
n_iter_w = max(len(hist_w["func"]) - 1, 1)
calls_w = (oracle_w.func_calls + oracle_w.grad_calls) / n_iter_w
t_w = float(np.asarray(hist_w["time"], dtype=float)[-1])

oracle_c = BealeOracle()
cautious_eps_beale = 1e-4
print(f"Используемый cautious_eps: {cautious_eps_beale}")
xf_c, msg_c, hist_c = cautious_lbfgs(
    oracle_c, x0_beale.copy(), tolerance=1e-8, max_iter=500, memory_size=10,
    line_search_options={"method": "Armijo", "c1": 1e-4, "alpha_0": 1.0},
    cautious_eps=cautious_eps_beale,
    trace=True, display=False,
)
n_iter_c = max(len(hist_c["func"]) - 1, 1)
calls_c = (oracle_c.func_calls + oracle_c.grad_calls) / n_iter_c
t_c = float(np.asarray(hist_c["time"], dtype=float)[-1])

skipped_total = int(hist_c["skipped_updates"][-1]) if hist_c.get("skipped_updates") else 0

print("Трек 4 (Бил): L-BFGS (Wolfe)")
print(f"  статус={msg_w}, итераций={n_iter_w}, время={t_w:.4f} с, среднее (func+grad)/итер ≈ {calls_w:.2f}")
print("Трек 4 (Бил): Cautious L-BFGS (Armijo)")
print(f"  статус={msg_c}, итераций={n_iter_c}, время={t_c:.4f} с, среднее (func+grad)/итер ≈ {calls_c:.2f}")
print(f"Трек 4 (Бил): пропущенных обновлений памяти: {skipped_total}")

g0w = hist_w["grad_norm"][0]
g0c = hist_c["grad_norm"][0]
tw = np.asarray(hist_w["time"], dtype=float)
tc = np.asarray(hist_c["time"], dtype=float)
rel_w = (np.asarray(hist_w["grad_norm"], dtype=float) / max(g0w, 1e-30)) ** 2
rel_c = (np.asarray(hist_c["grad_norm"], dtype=float) / max(g0c, 1e-30)) ** 2

fig, ax = plt.subplots()
ax.plot(tw, rel_w, lw=2, label="L-BFGS + Wolfe")
ax.plot(tc, rel_c, lw=2, label="Cautious L-BFGS + Armijo")
ax.set_yscale("log")
ax.set_xlabel("Время, с")
ax.set_ylabel(r"$(\Vert\nabla f(x_k)\Vert / \Vert\nabla f(x_0)\Vert)^2$")
ax.set_title(r"Трек 4: сходимость (функция Била, $x_0 \sim \mathcal{N}(0,10)$)")
ax.legend()
fig.savefig(FIGURES / "track4_comparison.png", bbox_inches="tight")
plt.show()

# Трек 4 (б): LogisticL2, phishing; отдельные оракулы для учёта вызовов
path_ph = require_file(ROOT / "data" / "phishing.txt")
X_ph, y_raw_ph = load_libsvm_dataset(path_ph)
n_ml = X_ph.shape[1]
x0_ml = RNG.normal(0.0, np.sqrt(10.0), size=n_ml)

oracle_lbfgs_ml, _ = make_logistic_oracle(X_ph, y_raw_ph)
xf_w_ml, msg_w_ml, hist_w_ml = lbfgs(
    oracle_lbfgs_ml, x0_ml.copy(), tolerance=TOLERANCE, max_iter=MAX_ITER, memory_size=10,
    line_search_options={"method": "Wolfe"}, trace=True, display=False,
)
n_iter_w_ml = max(len(hist_w_ml["func"]) - 1, 1)
calls_w_ml = (oracle_lbfgs_ml.func_calls + oracle_lbfgs_ml.grad_calls) / n_iter_w_ml
t_w_ml = float(np.asarray(hist_w_ml["time"], dtype=float)[-1])

oracle_caut_ml, _ = make_logistic_oracle(X_ph, y_raw_ph)
cautious_eps_ml = 0.1
print(f"Используемый cautious_eps: {cautious_eps_ml}")
xf_c_ml, msg_c_ml, hist_c_ml = cautious_lbfgs(
    oracle_caut_ml, x0_ml.copy(), tolerance=TOLERANCE, max_iter=MAX_ITER, memory_size=10,
    line_search_options={"method": "Armijo", "c1": 1e-4, "alpha_0": 1.0},
    cautious_eps=cautious_eps_ml,
    trace=True, display=False,
)
n_iter_c_ml = max(len(hist_c_ml["func"]) - 1, 1)
calls_c_ml = (oracle_caut_ml.func_calls + oracle_caut_ml.grad_calls) / n_iter_c_ml
t_c_ml = float(np.asarray(hist_c_ml["time"], dtype=float)[-1])
skipped_total_ml = int(hist_c_ml["skipped_updates"][-1]) if hist_c_ml.get("skipped_updates") else 0

print("Трек 4 (phishing, LogisticL2): L-BFGS (Wolfe)")
print(f"  статус={msg_w_ml}, итераций={n_iter_w_ml}, время={t_w_ml:.4f} с, среднее (func+grad)/итер ≈ {calls_w_ml:.2f}")
print("Трек 4 (phishing, LogisticL2): Cautious L-BFGS (Armijo)")
print(f"  статус={msg_c_ml}, итераций={n_iter_c_ml}, время={t_c_ml:.4f} с, среднее (func+grad)/итер ≈ {calls_c_ml:.2f}")
print(f"Трек 4 (phishing, ML): пропущенных обновлений: {skipped_total_ml}")

g0w_ml = hist_w_ml["grad_norm"][0]
g0c_ml = hist_c_ml["grad_norm"][0]
tw_ml = np.asarray(hist_w_ml["time"], dtype=float)
tc_ml = np.asarray(hist_c_ml["time"], dtype=float)
rel_w_ml = (np.asarray(hist_w_ml["grad_norm"], dtype=float) / max(g0w_ml, 1e-30)) ** 2
rel_c_ml = (np.asarray(hist_c_ml["grad_norm"], dtype=float) / max(g0c_ml, 1e-30)) ** 2

fig, ax = plt.subplots()
ax.plot(tw_ml, rel_w_ml, lw=2, label="L-BFGS + Wolfe")
ax.plot(tc_ml, rel_c_ml, lw=2, label="Cautious L-BFGS + Armijo")
ax.set_yscale("log")
ax.set_xlabel("Время, с")
ax.set_ylabel(r"$(\Vert\nabla f(x_k)\Vert / \Vert\nabla f(x_0)\Vert)^2$")
ax.set_title(r"Трек 4: LogisticL2, phishing, $x_0 \sim \mathcal{N}(0,10)$")
ax.legend()
fig.savefig(FIGURES / "track4_comparison_ml_phishing.png", bbox_inches="tight")
plt.show()
print("Трек 4: PNG сохранены:", FIGURES / "track4_comparison.png", ";", FIGURES / "track4_comparison_ml_phishing.png")

# %% [markdown]
# ## 8. Краткое резюме для отчёта

# %%
winner_lines = []
for slug in paths_24:
    best_m, best_t = None, float("inf")
    for m, hist in histories_24[slug].items():
        rgsq = relative_grad_sq(hist)
        idx = np.where(rgsq <= EPS_GRAD ** 2 + 1e-18)[0]
        if len(idx):
            t_stop = hist["time"][idx[0]]
            if t_stop < best_t:
                best_t, best_m = t_stop, m
    if best_m is None:
        winner_lines.append((slug, None, float("nan")))
    else:
        winner_lines.append((slug, METHOD_LABELS.get(best_m, best_m), best_t))

print("2.4 — самый быстрый до критерия (по wall-clock):")
for s, m, t in winner_lines:
    if m is None:
        print(f"   {s}: критерий не достигнут ни одним методом за MAX_ITER")
    else:
        print(f"   {s}: {m}, t ≈ {t:.4f} с")

o, l_, s = fracs["hfn"]
dom = "оракул" if o >= max(l_, s) else ("лин. алгебра" if l_ >= s else "линейный поиск")
print(f"2.5 — для HFN наибольшая средняя доля времени: {dom} (O={o:.3f}, L={l_:.3f}, LS={s:.3f})")


def last_improve_iter_classification(acc):
    acc = np.asarray(acc, dtype=float)
    best = acc[0]
    last = 0
    for i in range(1, len(acc)):
        if acc[i] > best + 1e-7:
            best = acc[i]
            last = i
    return last, best


def last_improve_iter_regression(mse):
    mse = np.asarray(mse, dtype=float)
    best = mse[0]
    last = 0
    for i in range(1, len(mse)):
        if mse[i] < best - 1e-8:
            best = mse[i]
            last = i
    return last, best


i_clf, v_clf = last_improve_iter_classification(hist26_clf["test_metric"])
i_reg, v_reg = last_improve_iter_regression(hist26_reg["test_metric"])
print(f"2.6 — phishing: последнее заметное улучшение Accuracy на итерации {i_clf} (значение ≈ {v_clf:.4f})")
print(f"2.6 — cadata: последнее заметное улучшение MSE на итерации {i_reg} (значение ≈ {v_reg:.4e})")

t_end_w = hist_w["time"][-1]
t_end_c = hist_c["time"][-1]
print("Трек 4 (функция Била):")
print(f"   Время до конца траектории: L-BFGS {t_end_w:.4f} с, Cautious {t_end_c:.4f} с")
print(f"   Средние вызовы (func+grad) на итерацию: L-BFGS {calls_w:.2f}, Cautious {calls_c:.2f}")
if calls_c < calls_w and t_end_c < t_end_w:
    print("   Вывод: отказ от Вульфа окупился и по времени, и по числу обращений к оракулу.")
elif calls_c < calls_w:
    print("   Вывод: по оракулу осторожный вариант выгоднее; по времени — см. цифры выше.")
elif t_end_c < t_end_w:
    print("   Вывод: по времени осторожный вариант быстрее; по оракулу — см. цифры.")
else:
    print("   Вывод: преимущество по времени/оракулу у классического L-BFGS на этом запуске.")

t_end_w_ml = hist_w_ml["time"][-1]
t_end_c_ml = hist_c_ml["time"][-1]
print("Трек 4 (phishing, LogisticL2):")
print(f"   Время до конца траектории: L-BFGS {t_end_w_ml:.4f} с, Cautious {t_end_c_ml:.4f} с")
print(f"   Средние вызовы (func+grad) на итерацию: L-BFGS {calls_w_ml:.2f}, Cautious {calls_c_ml:.2f}")
if calls_c_ml < calls_w_ml and t_end_c_ml < t_end_w_ml:
    print("   Вывод (ML): отказ от Вульфа окупился и по времени, и по оракулу.")
elif calls_c_ml < calls_w_ml:
    print("   Вывод (ML): по оракулу осторожный вариант выгоднее; по времени — см. цифры выше.")
elif t_end_c_ml < t_end_w_ml:
    print("   Вывод (ML): по времени осторожный вариант быстрее; по оракулу — см. цифры.")
else:
    print("   Вывод (ML): сравните цифры с классическим L-BFGS на этом датасете и инициализации.")
