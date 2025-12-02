# tsunami_model_pipeline.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# 1) 데이터 로드
csv_path = "earthquake_data_tsunami.csv"
df = pd.read_csv(csv_path)
print("데이터 샘플:")
print(df.head())

# 2) 컬럼 설정 - 필요에 따라 여기 수정하세요
# 예시: 수치형(feature)과 범주형(feature) 분리
numeric_features = ['magnitude', 'depth_km', 'distance_to_coast_km']  # CSV에 없는 컬럼은 제거/수정하세요
categorical_features = []  # 예: ['focal_mechanism']

label_col = 'tsunami'  # 0/1

# 간단한 확인
for c in numeric_features:
    if c not in df.columns:
        print(f"경고: '{c}' 컬럼이 데이터에 없습니다. feature list를 확인하세요.")

if label_col not in df.columns:
    raise KeyError(f"라벨 컬럼 '{label_col}'이 CSV에 없습니다. 파일을 확인하세요.")

X = df[numeric_features + categorical_features].copy()
y = df[label_col].astype(int)

# 3) 전처리 및 파이프라인
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'  # 나머지 컬럼 드랍
)

# 4) 모델 + 전체 파이프라인
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', clf)])

# 5) 학습/검증 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("학습샘플:", X_train.shape, "테스트샘플:", X_test.shape)

# 6) (선택) 하이퍼파라미터 탐색 - 시간이 걸립니다.
do_grid_search = True
if do_grid_search:
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print("베스트 파라미터:", grid.best_params_)
    best_model = grid.best_estimator_
else:
    pipe.fit(X_train, y_train)
    best_model = pipe

# 7) 평가
y_pred = best_model.predict(X_test)
if hasattr(best_model, "predict_proba"):
    y_proba = best_model.predict_proba(X_test)[:,1]
else:
    y_proba = None

print("정확도:", accuracy_score(y_test, y_pred))
print("분류 리포트:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("혼동 행렬:\n", cm)

if y_proba is not None:
    auc = roc_auc_score(y_test, y_proba)
    print("ROC AUC:", auc)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.show()

# 8) 중요 변수 확인 (RandomForest의 경우)
try:
    feature_names = numeric_features + list(best_model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)) if categorical_features else numeric_features
except Exception:
    feature_names = numeric_features  # fallback

try:
    importances = best_model.named_steps['classifier'].feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("Feature importances:\n", fi)
    plt.figure(figsize=(6,4))
    sns.barplot(x=fi.values, y=fi.index)
    plt.title("Feature importances")
    plt.show()
except Exception as e:
    print("피처 중요도 확인에 실패했습니다:", e)

# 9) 모델 저장
model_path = "tsunami_random_forest.joblib"
joblib.dump(best_model, model_path)
print(f"학습된 모델을 '{model_path}'로 저장했습니다.")

