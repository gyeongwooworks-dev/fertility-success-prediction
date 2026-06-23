import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# ============================================================
# 전처리에 사용할 상수 정의
# ============================================================

COLS_TO_DROP = [
    'ID', '임신 시도 또는 마지막 임신 경과 연수', '난자 해동 경과일',
    '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제',
    '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성',
    '불임 원인 - 정자 농도', '불임 원인 - 정자 형태',
    'PGS 시술 여부', 'PGD 시술 여부', '착상 전 유전 검사 사용 여부',
]

# 나이 구간을 대표 수치와 임상 위험군으로 변환하는 매핑
# Normal        : 자연 임신 성공률이 양호한 정상군
# High_Early    : 만 35세 이상, 난소 예비능 저하가 시작되는 시기
# High_Extreme  : 만 40세 이상, 자가 난자 임신 성공률이 급격히 떨어지는 구간
# High_Reversal : 만 45세 이상, 기증 난자 사용으로 일부 역전이 관찰되는 구간
AGE_INFO = {
    '만18-34세': {'median': 26.0, 'risk': 'Normal'},
    '만35-37세': {'median': 36.0, 'risk': 'High_Early'},
    '만38-39세': {'median': 38.5, 'risk': 'High_Early'},
    '만40-42세': {'median': 41.0, 'risk': 'High_Extreme'},
    '만43-44세': {'median': 43.5, 'risk': 'High_Extreme'},
    '만45-50세': {'median': 47.5, 'risk': 'High_Reversal'},
    '알 수 없음': {'median': None, 'risk': 'Unknown'},
}

# 나이 결측 시 사용할 보정값
# 38.5는 전체 분포의 중앙값에 가까운 수치로, 통계 학습이 아닌 도메인 판단에 따른 고정값
AGE_MEDIAN_FILLNA = 38.5

# '배아 생성 주요 이유' 멀티라벨 컬럼 처리에 사용할 알려진 클래스 목록
# 학습 데이터에서 이미 알고 있는 클래스를 하드코딩하여,
# 평가 데이터에서 새로운 값이 들어오더라도 동일한 5개 컬럼만 생성되도록 보장
MULTILABEL_COL = '배아 생성 주요 이유'
KNOWN_REASONS  = ['현재 시술용', '배아 저장용', '난자 저장용', '기증용', '연구용']


# ============================================================
# 전처리 함수 정의
# ============================================================

def engineer_features(df, df_raw_ref=None):
    """
    도메인 지식 기반의 파생 변수들을 생성합니다.

    매개변수
        df          : 변환을 적용할 데이터프레임 (이미 COLS_TO_DROP이 적용된 상태)
        df_raw_ref  : 원본 데이터프레임 참조. PGS/PGD 컬럼은 COLS_TO_DROP에서
                      제거되었기 때문에, 유전검사_시행여부 변수를 만들기 위해
                      원본 데이터를 별도로 받아 사용합니다.
                      df_raw_ref와 df는 행 순서가 동일해야 합니다.
    """
    df = df.copy()

    # 1. 나이 변수 가공
    # 원본의 '시술 당시 나이'는 구간 문자열 → 대표 수치(Age_Median)와 임상 위험군(Age_Risk_Group)으로 분리
    if '시술 당시 나이' in df.columns:
        df['Age_Median'] = df['시술 당시 나이'].map(
            lambda x: AGE_INFO.get(x, {'median': None})['median'])
        df['Age_Risk_Group'] = df['시술 당시 나이'].map(
            lambda x: AGE_INFO.get(x, {'risk': 'Unknown'})['risk'])
        df = df.drop(columns=['시술 당시 나이'])

    # 2. 유전검사 시행 여부 통합
    # PGS와 PGD는 모두 착상 전 유전 진단의 하위 개념.
    # 둘 중 어느 하나라도 시행된 경우 '유전검사_시행여부'를 1로 표시.
    # 원본 컬럼은 COLS_TO_DROP에서 제거되므로 df_raw_ref에서 직접 참조.
    if df_raw_ref is not None:
        df['유전검사_시행여부'] = (
            df_raw_ref['PGS 시술 여부'].notna() |
            df_raw_ref['PGD 시술 여부'].notna()
        ).astype(int).values

    # 3. 기증 난자 관련 상호작용 변수
    # 만 45-50세 구간에서 임신 성공률이 일부 회복되는 현상은 기증 난자 사용 때문.
    # 자가 난자일 때만 나이 페널티가 작동하는 변수를 별도로 생성.
    #   기증난자×나이 : 기증 난자를 받은 경우에만 값이 살아 있고, 나머지는 0
    #   자가난자×나이 : 자가 난자인 경우에만 값이 살아 있고, 나머지는 0
    if '난자 출처' in df.columns:
        df['기증난자_여부'] = (df['난자 출처'] == '기증 제공').astype(int)
        age_median = df['Age_Median'].fillna(47.5)
        df['기증난자×나이'] = df['기증난자_여부']      * age_median
        df['자가난자×나이'] = (1 - df['기증난자_여부']) * age_median

    # 4. 배아 관련 파생 변수
    # 배아_잉여율 : (총 생성 - 이식)/총 생성. 값이 높다는 것은 양질의 배아가 충분히 만들어졌다는 신호.
    # Age×배아수  : 나이와 총 배아 수의 곱. 고령에서 배아가 많을 때의 효과를 별도로 표현.
    total    = df.get('총 생성 배아 수', pd.Series(0, index=df.index))
    transfer = df.get('이식된 배아 수',  pd.Series(0, index=df.index))
    df['배아_잉여율'] = np.where(total > 0, (total - transfer) / total, 0).clip(0, 1)
    df['Age×배아수'] = df['Age_Median'].fillna(47.5) * total

    # 5. 수정률 (정자 품질의 간접 지표)
    # 채취된 신선 난자 중 실제로 수정된 비율. 낮다면 정자 운동성/농도 문제 가능성.
    collected = df.get('수집된 신선 난자 수', pd.Series(0, index=df.index))
    mixed     = df.get('혼합된 난자 수',      pd.Series(0, index=df.index))
    df['수정률'] = np.where(collected > 0, mixed / collected, 0).clip(0, 1)

    # 6. 저장 배아 보유 여부 (이진 플래그)
    # 저장된 배아가 있다는 것은 다음 시도를 위한 자원이 있다는 뜻.
    stored = df.get('저장된 배아 수', pd.Series(0, index=df.index))
    df['저장배아_보유여부'] = (stored > 0).astype(int)

    # 7. 시술 횟수 통합
    # IVF와 DI 시술 횟수를 합쳐 총 시술 횟수를 계산.
    # 횟수 컬럼은 '0회', '1회'와 같은 문자열이므로 정규식으로 숫자만 추출.
    def parse_count(col_name):
        s = df.get(col_name, pd.Series('0회', index=df.index))
        return s.astype(str).str.extract(r'(\d+)')[0].fillna(0).astype(int)
    df['총_시술횟수'] = parse_count('IVF 시술 횟수') + parse_count('DI 시술 횟수')

    # 8. 기증 정자 사용 여부
    # '기증자 정자와 혼합된 난자 수' 컬럼은 기증 정자를 사용하지 않았을 때 결측으로 기록.
    # 즉 '결측이 아니다'라는 것 자체가 기증 정자를 사용했다는 신호.
    df['기증정자_사용여부'] = df.get(
        '기증자 정자와 혼합된 난자 수',
        pd.Series(np.nan, index=df.index)
    ).notna().astype(int)

    return df


def encode_multilabel(df):
    """
    '배아 생성 주요 이유' 컬럼은 한 행에 여러 값이 콤마로 구분된 멀티라벨 변수.
    MultiLabelBinarizer에 classes를 KNOWN_REASONS로 하드코딩하여,
    학습/평가 데이터가 항상 같은 5개 컬럼을 생성하도록 보장.
    """
    df = df.copy()
    if MULTILABEL_COL not in df.columns:
        return df

    split_series = df[MULTILABEL_COL].fillna('').apply(
        lambda x: [v.strip() for v in x.split(',') if v.strip()]
    )
    mlb = MultiLabelBinarizer(classes=KNOWN_REASONS)
    encoded = pd.DataFrame(
        mlb.fit_transform(split_series),
        columns=[f'목적_{c}' for c in KNOWN_REASONS],
        index=df.index
    )
    return pd.concat([df.drop(columns=[MULTILABEL_COL]), encoded], axis=1)


def preprocess_missing(df, target=None):
    """
    결측치를 고정 상수로 보정합니다.
      수치형 결측 → 0
      범주형 결측 → '알 수 없음'

    통계 기반 보정(평균/중앙값)을 사용하지 않는 이유:
    K-Fold 검증에서 fold 간 정보 유출의 빌미가 될 수 있기 때문.
    """
    df = df.copy()
    if target and target in df.columns:
        y = df.pop(target)
    else:
        y = None

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(exclude='object').columns.tolist()
    df[cat_cols] = df[cat_cols].fillna('알 수 없음')
    df[num_cols] = df[num_cols].fillna(0)

    if y is not None:
        df[target] = y.values
    return df


def build_features(df, df_raw_ref=None, target=None):
    """
    전처리 3단계를 순서대로 호출하는 통합 함수.
      1) engineer_features  : 도메인 기반 파생 변수 생성
      2) encode_multilabel  : 멀티라벨을 0/1 컬럼으로 분해
      3) preprocess_missing : 결측 보정
      4) 범주형 컬럼을 category dtype으로 변환 (LightGBM, CatBoost 내부 처리)
    """
    df = engineer_features(df, df_raw_ref=df_raw_ref)
    df = encode_multilabel(df)
    df = preprocess_missing(df, target=target)

    cat_cols = [c for c in df.select_dtypes(include='object').columns if c != target]
    for col in cat_cols:
        df[col] = df[col].astype('category')

    if target and target in df.columns:
        return df.drop(columns=[target]), df[target]
    return df, None


def preprocess(df_train_raw, df_test_raw, target='임신 성공 여부'):
    """
    학습/평가 데이터에 전처리 파이프라인을 적용하는 최상위 함수.

    반환
        X               : 학습 피처 데이터프레임
        y               : 학습 타겟 시리즈
        X_test          : 평가 피처 데이터프레임
        scale_pos_weight: LightGBM/CatBoost 클래스 불균형 보정 가중치
    """
    print('전처리 파이프라인을 시작합니다.')

    # 1. 불필요한 컬럼 제거
    df      = df_train_raw.drop(columns=COLS_TO_DROP, errors='ignore').copy()
    df_test = df_test_raw.drop(columns=COLS_TO_DROP,  errors='ignore').copy()

    # 2. 피처 빌드 (학습: X+y, 평가: X_test)
    X, y      = build_features(df,      df_raw_ref=df_train_raw, target=target)
    X_test, _ = build_features(df_test, df_raw_ref=df_test_raw,  target=None)

    # 3. 학습 데이터에만 존재하는 컬럼 처리
    for col in set(X.columns) - set(X_test.columns):
        X_test[col] = 0
    X_test = X_test[X.columns]

    # 4. 카테고리 dtype 일관성 보장
    cat_cols = [c for c in X.columns if X[c].dtype.name == 'category']
    for col in cat_cols:
        X_test[col] = X_test[col].astype(str).astype(
            pd.CategoricalDtype(categories=X[col].cat.categories)
        )

    # 5. 클래스 불균형 보정 가중치
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    print(f'  학습 피처 수: {X.shape[1]}개  |  평가 피처 수: {X_test.shape[1]}개')
    print(f'  컬럼 일치 여부: {list(X.columns) == list(X_test.columns)}')
    print(f'  클래스 불균형 보정 가중치: {scale_pos_weight:.4f}')
    return X, y, X_test, scale_pos_weight
