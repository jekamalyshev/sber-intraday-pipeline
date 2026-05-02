"""Walk-forward валидация с purged CV + embargo для CatBoost + ATR-target k_bars=3.

Цель: проверить, реальный ли edge AUC OOS≈0.60 на одиночном split, или это
"удачный кусок". Делаем 5 фолдов expanding-window с покупкой между train/test
зазора (embargo) для предотвращения утечки через target, который смотрит
на k_bars свечей вперёд.

Схема:
  - Все доступные строки X,y (после фильтрации NaN/corr) делим на N+1 блок.
  - На каждом фолде i ∈ [1..N]:
      train = блоки [0..i-1]  (expanding)
      embargo = K_BARS строк сразу после train (исключаем)
      test = блок i (фиксированной длины)
  - На train-блоке внутри ещё один split на (sub-train, sub-valid)
    для early stopping CatBoost.
  - Метрики считаем raw на test, без отдельной калибровки (Platt тут не
    нужен для proper edge-теста — он влияет только на пороги, не на AUC).

Воспроизводит ту же сборку фичей, что в основном ноутбуке.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             roc_auc_score)

warnings.filterwarnings('ignore')

# ── Загружаем функции из ноутбука через exec ────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / 'sber_intraday_pipeline.ipynb'

import nbformat


def _load_notebook_helpers():
    """Выполняем helper-ячейки ноутбука, чтобы получить prepare_ohlcv_dataframe,
    add_*_features, build_feature_dataframe, build_X_y_for_model и др.

    Берём только ячейки с определениями (def/class/import) — пропускаем те,
    что выполняют тяжёлый код или строят графики.
    """
    nb = nbformat.read(str(NB_PATH), as_version=4)
    ns: dict = {}
    # Выполняем только helper-ячейки, в которых нет реального запуска модели:
    # 2 (imports), 4 (series_to_supervised), 6 (prepare/add_domain/rolling/calendar),
    # 7-8 (TA, build_feature_dataframe), 9 (build_X_y_for_model)
    helper_indices = [2, 4, 6, 7, 8, 9]
    for idx in helper_indices:
        if idx >= len(nb.cells):
            continue
        cell = nb.cells[idx]
        if cell.cell_type != 'code':
            continue
        src = cell.source
        # Удаляем magic-команды, которые не работают вне jupyter
        src = '\n'.join(
            line for line in src.split('\n')
            if not line.strip().startswith('%')
        )
        try:
            exec(src, ns)
        except Exception as e:
            print(f'[helpers] cell {idx}: skipped due to {type(e).__name__}: {e}')
    return ns


HELPERS = _load_notebook_helpers()
build_feature_dataframe = HELPERS['build_feature_dataframe']
build_X_y_for_model = HELPERS['build_X_y_for_model']

# Параметры таргета — те же, что в основной конфигурации
import builtins
builtins.K_BARS = 3
builtins.K_ATR = 1.0


# ── Walk-forward CV ─────────────────────────────────────────────────────────

def purged_walk_forward_splits(n_samples: int, n_splits: int, embargo: int):
    """Генератор (train_idx, test_idx) для expanding-window walk-forward
    с embargo между train и test.

    Делим хвост ряда (после первого минимального train-блока) на n_splits равных
    частей и каждый раз тестируемся на следующей части, добавляя предыдущие
    в train.
    """
    # Минимальный размер первого train-блока — половина данных
    min_train = n_samples // 2
    test_pool = n_samples - min_train
    fold_size = test_pool // n_splits
    for i in range(n_splits):
        test_start = min_train + i * fold_size
        test_end = test_start + fold_size if i < n_splits - 1 else n_samples
        train_end = test_start - embargo  # purge: убираем embargo строк перед test
        if train_end <= 0:
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        yield i + 1, train_idx, test_idx


def fit_predict_catboost(X_tr, y_tr, X_te, y_te, X_va=None, y_va=None):
    """Обучаем CatBoost с теми же гиперпараметрами, что в основном ноутбуке.

    Если X_va/y_va не переданы — отрезаем последние 15% train под валидацию
    для early stopping (тоже хронологически).
    """
    from catboost import CatBoostClassifier

    if X_va is None:
        n = len(X_tr)
        cut = int(n * 0.85)
        X_va, y_va = X_tr.iloc[cut:], y_tr.iloc[cut:]
        X_tr, y_tr = X_tr.iloc[:cut], y_tr.iloc[:cut]

    clf = CatBoostClassifier(
        iterations=500, depth=5, learning_rate=0.05,
        l2_leaf_reg=3.0, subsample=0.8, rsm=0.7,
        bootstrap_type='Bernoulli', auto_class_weights='Balanced',
        eval_metric='Logloss', od_type='Iter', od_wait=30,
        random_seed=42, thread_count=-1,
        verbose=False, allow_writing_files=False,
    )
    clf.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        'auc': roc_auc_score(y_te, proba),
        'acc': accuracy_score(y_te, pred),
        'logloss': log_loss(y_te, proba, labels=[0, 1]),
        'brier': brier_score_loss(y_te, proba),
        'best_iter': clf.get_best_iteration(),
        'n_train': len(X_tr),
        'n_valid': len(X_va),
        'n_test': len(X_te),
    }


def main():
    print('=' * 70)
    print('Walk-forward CV: CatBoost + ATR-target (k_bars=3, k_atr=1.0)')
    print('=' * 70)

    # ── 1. Сборка фичей и таргета (как в ноутбуке) ──────────────────────────
    t0 = time.time()
    DATA_PATH = ROOT / 'Сбербанк' / 'year_result.csv'
    print(f'\n[1] Reading {DATA_PATH}...')
    frame = pd.read_csv(DATA_PATH, header=0, sep=';')
    frame.columns = [c.strip('<>').strip() for c in frame.columns]
    print(f'    rows={len(frame):,}')

    print('\n[2] Building features (build_feature_dataframe)...')
    feature_df = build_feature_dataframe(frame)
    print(f'    feature_df shape: {feature_df.shape}')

    print('\n[3] Building supervised X,y (build_X_y_for_model, n_in=3)...')
    X, y, leakage_cols = build_X_y_for_model(feature_df, n_in=3)
    print(f'    X shape: {X.shape} | y shape: {y.shape}')
    print(f'    Class balance: {dict(y.value_counts(normalize=True).round(3))}')

    # Чистим inf/nan на всякий случай (как в timeframe_test)
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median()
    X = X.fillna(medians).fillna(0)

    elapsed = time.time() - t0
    print(f'    [feature build done in {elapsed:.1f}s]')

    # ── 2. Walk-forward CV ─────────────────────────────────────────────────
    N_SPLITS = 5
    EMBARGO = builtins.K_BARS  # 3 свечи — горизонт target'а
    n_samples = len(X)
    print(f'\n[4] Walk-forward: {N_SPLITS} splits, embargo={EMBARGO} строк')
    print(f'    n_samples={n_samples}, min_train={n_samples//2}')

    fold_results = []
    for fold_id, train_idx, test_idx in purged_walk_forward_splits(
        n_samples, N_SPLITS, EMBARGO
    ):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        print(f'\n  Fold {fold_id}/{N_SPLITS}: '
              f'train=[0..{train_idx[-1]}] (n={len(train_idx)}) | '
              f'test=[{test_idx[0]}..{test_idx[-1]}] (n={len(test_idx)})')
        t1 = time.time()
        m = fit_predict_catboost(X_tr, y_tr, X_te, y_te)
        m['fold'] = fold_id
        m['train_end_idx'] = int(train_idx[-1])
        m['test_start_idx'] = int(test_idx[0])
        m['test_end_idx'] = int(test_idx[-1])
        m['fit_time_sec'] = round(time.time() - t1, 1)
        # Class balance в test для контекста
        m['test_pos_rate'] = round(float(y_te.mean()), 3)
        fold_results.append(m)
        print(f'    AUC={m["auc"]:.4f} | Acc={m["acc"]:.4f} | '
              f'LogLoss={m["logloss"]:.4f} | best_iter={m["best_iter"]} | '
              f'time={m["fit_time_sec"]}s | pos_rate={m["test_pos_rate"]}')

    # ── 3. Агрегаты ────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    aucs = [m['auc'] for m in fold_results]
    accs = [m['acc'] for m in fold_results]
    lls = [m['logloss'] for m in fold_results]
    summary = {
        'n_splits_completed': len(fold_results),
        'embargo_bars': EMBARGO,
        'auc_mean': round(float(np.mean(aucs)), 4),
        'auc_std': round(float(np.std(aucs)), 4),
        'auc_min': round(float(np.min(aucs)), 4),
        'auc_max': round(float(np.max(aucs)), 4),
        'auc_median': round(float(np.median(aucs)), 4),
        'acc_mean': round(float(np.mean(accs)), 4),
        'acc_std': round(float(np.std(accs)), 4),
        'logloss_mean': round(float(np.mean(lls)), 4),
        'reference_single_split_auc_calib': 0.6004,
    }
    print(f"\nAUC per fold: {[round(a, 4) for a in aucs]}")
    print(f"AUC mean ± std: {summary['auc_mean']} ± {summary['auc_std']}")
    print(f"AUC min/median/max: "
          f"{summary['auc_min']} / {summary['auc_median']} / {summary['auc_max']}")
    print(f"Reference (single-split Calib): {summary['reference_single_split_auc_calib']}")

    # ── 4. Запись результатов ──────────────────────────────────────────────
    out = {
        'config': {
            'model': 'CatBoost',
            'k_bars': builtins.K_BARS,
            'k_atr': builtins.K_ATR,
            'n_splits': N_SPLITS,
            'embargo': EMBARGO,
            'n_in': 3,
        },
        'folds': fold_results,
        'summary': summary,
    }
    out_path = ROOT / 'scripts' / 'walk_forward_cv_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n[saved] {out_path}')


if __name__ == '__main__':
    main()
