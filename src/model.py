import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ============================================================
# LightGBM 하이퍼파라미터
# ============================================================

# v4 파라미터: Optuna 탐색으로 얻은 최종 파라미터 (LB 0.74191 기록)
v4_params = {
    'learning_rate':     0.032275921555211334,
    'num_leaves':        32,
    'min_child_samples': 31,
    'subsample':         0.9443563832121702,
    'subsample_freq':    1,
    'colsample_bytree':  0.4660046628277167,
    'reg_alpha':         3.546463779196944,
    'reg_lambda':        2.1863992476849217,
}

# day4_5 파라미터: Optuna 100회 탐색에서 얻어진 더 보수적인 모델
# 76개 피처 환경에서 탐색되었기에 71개 피처 환경에서는 결과가 다를 수 있음
day45_params = {
    'learning_rate':     0.01227,
    'num_leaves':        18,
    'min_child_samples': 31,
    'subsample':         0.9443563832121702,
    'subsample_freq':    1,
    'colsample_bytree':  0.4660046628277167,
    'reg_alpha':         3.546463779196944,
    'reg_lambda':        8.839,
}


# ============================================================
# 학습 함수 정의
# ============================================================

def run_lgbm_oof(X, y, X_test, params, n_splits=5, tag='lgbm'):
    """
    LightGBM을 Stratified K-Fold로 학습하고 OOF 예측 및 평가 데이터 예측을 반환합니다.

    반환
        models     : fold별 학습된 LGBMClassifier 리스트
        oof_preds  : 학습 데이터 전체에 대한 OOF 예측 확률 배열
        test_preds : 평가 데이터에 대한 K-fold 평균 예측 확률 배열
        oof_auc    : 전체 OOF 예측으로 계산한 AUC
    """
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    params = {**params,
              'scale_pos_weight': scale_pos_weight,
              'objective':        'binary',
              'metric':           'auc',
              'random_state':     42,
              'n_jobs':           -1,
              'verbose':          -1}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds  = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))
    cat_feats  = [c for c in X.columns if X[c].dtype.name == 'category']
    models, fold_aucs = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(500)],
            categorical_feature=cat_feats,
        )
        oof_preds[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        test_preds        += model.predict_proba(X_test)[:, 1] / n_splits

        auc = roc_auc_score(y.iloc[val_idx], oof_preds[val_idx])
        fold_aucs.append(auc)
        print(f'  [{tag}] Fold {fold+1}/{n_splits}: AUC={auc:.4f}  iter={model.best_iteration_}')
        models.append(model)

    oof_auc = roc_auc_score(y, oof_preds)
    print(f'\n  [{tag}] OOF AUC={oof_auc:.4f}  (folds={n_splits})')
    return models, oof_preds, test_preds, oof_auc


def run_catboost_oof(X, y, X_test, n_splits=5, use_gpu=False):
    """
    CatBoost를 동일한 K-Fold 구조로 학습합니다.
    CatBoost는 카테고리를 인덱스 형태로 받기 때문에 dtype을 string으로 변환합니다.

    반환
        models     : fold별 학습된 CatBoostClassifier 리스트
        oof_preds  : 학습 데이터 전체에 대한 OOF 예측 확률 배열
        test_preds : 평가 데이터에 대한 K-fold 평균 예측 확률 배열
        oof_auc    : 전체 OOF 예측으로 계산한 AUC
    """
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    cat_feats_idx = [i for i, c in enumerate(X.columns) if X[c].dtype.name == 'category']

    # CatBoost는 category dtype을 직접 다루지 못하므로 string으로 변환
    X_cb      = X.copy()
    X_test_cb = X_test.copy()
    for col in X.columns:
        if X[col].dtype.name == 'category':
            X_cb[col]      = X_cb[col].astype(str)
            X_test_cb[col] = X_test_cb[col].astype(str)

    # CatBoost 하이퍼파라미터: 17번 실험에서 기본값이 가장 안정적이었음
    params = {
        'iterations':          2000,
        'learning_rate':       0.05,
        'depth':               6,
        'l2_leaf_reg':         3,
        'scale_pos_weight':    scale_pos_weight,
        'eval_metric':         'AUC',
        'random_seed':         42,
        'verbose':             False,
        'early_stopping_rounds': 100,
    }
    if use_gpu:
        params['task_type'] = 'GPU'
        params['devices']   = '0'

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds  = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))
    models, fold_aucs = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_cb, y)):
        model = CatBoostClassifier(**params)
        model.fit(
            X_cb.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=(X_cb.iloc[val_idx], y.iloc[val_idx]),
            cat_features=cat_feats_idx,
        )
        oof_preds[val_idx]  = model.predict_proba(X_cb.iloc[val_idx])[:, 1]
        test_preds         += model.predict_proba(X_test_cb)[:, 1] / n_splits
        auc = roc_auc_score(y.iloc[val_idx], oof_preds[val_idx])
        fold_aucs.append(auc)
        print(f'  [catboost] Fold {fold+1}/{n_splits}: AUC={auc:.4f}')
        models.append(model)

    oof_auc = roc_auc_score(y, oof_preds)
    print(f'\n  [catboost] OOF AUC={oof_auc:.4f}  (folds={n_splits})')
    return models, oof_preds, test_preds, oof_auc


def to_rank(arr):
    """
    예측 확률 배열을 [0, 1] 범위의 순위로 변환합니다.
    두 모델의 예측을 평균낼 때 서로 다른 확률 분포를 동일한 척도로 맞춰주어
    분포 차이로 인한 왜곡을 줄여 줍니다.
    """
    from scipy.stats import rankdata
    return rankdata(arr) / len(arr)


def weighted_ensemble(oof_list, test_list, y, weights=None, use_rank=False, names=None):
    """
    여러 모델의 OOF 예측과 평가 예측을 가중 평균하는 앙상블 함수입니다.

    매개변수
        oof_list  : 각 모델의 OOF 예측 배열 리스트
        test_list : 각 모델의 평가 예측 배열 리스트
        y         : 실제 타겟 (가중치 자동 계산 및 AUC 산출에 사용)
        weights   : 직접 지정할 가중치 (None이면 OOF AUC 비례 자동 계산)
        use_rank  : True이면 평균 전에 순위 변환 적용 (Rank Averaging)
        names     : 모델 이름 리스트 (로그 출력용)

    반환
        oof_ensemble  : 앙상블 OOF 예측
        test_ensemble : 앙상블 평가 예측
    """
    if use_rank:
        oof_list_proc  = [to_rank(o) for o in oof_list]
        test_list_proc = [to_rank(t) for t in test_list]
        print('[앙상블] Rank Averaging 모드')
    else:
        oof_list_proc  = oof_list
        test_list_proc = test_list
        print('[앙상블] 확률값 평균 모드')

    if weights is None:
        aucs   = [roc_auc_score(y, oof) for oof in oof_list]
        total  = sum(aucs)
        weights = [a / total for a in aucs]
        if names:
            for n, w, a in zip(names, weights, aucs):
                print(f'  {n}: weight={w:.4f}, OOF AUC={a:.4f}')

    oof_ensemble  = sum(w * o for w, o in zip(weights, oof_list_proc))
    test_ensemble = sum(w * t for w, t in zip(weights, test_list_proc))

    ens_auc = roc_auc_score(y, oof_ensemble)
    print(f'[앙상블] 최종 OOF AUC={ens_auc:.4f}')
    return oof_ensemble, test_ensemble
