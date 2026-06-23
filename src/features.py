import pandas as pd


def add_team_features(X_input):
    """
    팀원이 설계한 임상 도메인 피처 5개를 추가합니다.

    매개변수
        X_input : 전처리가 완료된 데이터프레임 (preprocessing.preprocess 반환값)

    반환
        새 변수 5개가 추가된 데이터프레임
    """
    df = X_input.copy()

    # 변수 1: 배반포_5일차_이식
    # 배반포(blastocyst) 단계는 수정 후 5~6일차의 배아 발달 단계.
    # 이 단계까지 키워서 이식하는 방식은 학계에서 일관되게 우위가 보고됨 (Roque 2013).
    # 정의: 배아 이식 경과일이 정확히 5일이고, 실제로 이식이 진행된 경우 1
    df['배반포_5일차_이식'] = (
        (df['배아 이식 경과일'] == 5) &
        (df['이식된 배아 수'].fillna(0) > 0)
    ).astype(int)

    # 변수 2: 젊은_고효율_이식
    # 젊은 환자 + 고품질 배아 + 실제 이식 = 임상적으로 최적의 시술 조건.
    # 정의: Age_Median ≤ 37 + 배아_잉여율 > 0.3 + 이식된 배아 수 ≥ 1
    df['젊은_고효율_이식'] = (
        (df['Age_Median'].fillna(47.5) <= 37) &
        (df['배아_잉여율'].fillna(0) > 0.3) &
        (df['이식된 배아 수'].fillna(0) >= 1)
    ).astype(int)

    # 변수 3: 이식경과일_구간
    # 배아 이식 경과일은 본질적으로 구간형 정보.
    # 수치형 처리만으로는 잡히지 않는 신호를 구간화로 보강.
    #   no_transfer : 결측 (이식이 진행되지 않음)
    #   early       : 3일 이하 (분할기 단계 이식)
    #   blastocyst  : 정확히 5일 (배반포 이식)
    #   other       : 그 외 (4일 또는 6일 이상)
    def categorize_transfer_day(x):
        if pd.isna(x):
            return 'no_transfer'
        elif x <= 3:
            return 'early'
        elif x == 5:
            return 'blastocyst'
        else:
            return 'other'

    df['이식경과일_구간'] = df['배아 이식 경과일'].apply(categorize_transfer_day).astype('category')

    # 변수 4: 동결_기증_복합
    # 만 45-50세 구간에서 임신 성공률이 일부 회복되는 현상은
    # '기증 난자 + 동결 배아' 조합의 사용 증가가 핵심 원인 (Cozzolino 2024).
    # 정의: 동결 배아 사용 = 1 + 기증난자 = 1
    df['동결_기증_복합'] = (
        (df.get('동결 배아 사용 여부', pd.Series(0, index=df.index)).fillna(0) == 1) &
        (df['기증난자_여부'] == 1)
    ).astype(int)

    # 변수 5: 고령_반복시술
    # 고령 + 반복 시술 = 임상적으로 가장 어려운 환자군.
    # 정의: Age_Median ≥ 40 + 총_시술횟수 ≥ 2
    df['고령_반복시술'] = (
        (df['Age_Median'].fillna(47.5) >= 40) &
        (df['총_시술횟수'].fillna(0) >= 2)
    ).astype(int)

    return df


def align_team_feature_categories(X, X_test):
    """
    add_team_features 적용 후 카테고리형 컬럼('이식경과일_구간')의
    dtype을 학습/평가 데이터 간에 일치시킵니다.

    매개변수
        X      : add_team_features가 적용된 학습 데이터프레임
        X_test : add_team_features가 적용된 평가 데이터프레임

    반환
        카테고리 dtype이 정렬된 X_test
    """
    for col in ['이식경과일_구간']:
        if col in X.columns and col in X_test.columns:
            X_test[col] = X_test[col].astype(str).astype(
                pd.CategoricalDtype(categories=X[col].cat.categories)
            )
    return X_test
