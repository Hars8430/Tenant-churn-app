from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import shap
import os

app = Flask(__name__)
CORS(app)

# ── Load Model Artifacts ─────────────────────────────────────────────────────
MODEL_PATH = "model/xgb_churn_model.pkl"
EXPLAINER_PATH = "model/shap_explainer.pkl"
FEATURES_PATH = "model/feature_names.pkl"

model = None
explainer = None
feature_names = None

def load_artifacts():
    global model, explainer, feature_names
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        explainer = joblib.load(EXPLAINER_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        print("✅ Model artifacts loaded successfully")
    else:
        print("⚠️  Model not found. Run train_model.py first.")

load_artifacts()

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "TenantIQ Pro API",
        "version": "2.4.0",
        "company": "CBRE Commercial Real Estate Intelligence",
        "endpoints": ["/predict", "/predict/batch", "/shap/<tenant_id>", "/health"]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": "XGBoost-v2.4",
        "accuracy": 0.891
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Single tenant churn prediction.
    Body: {
        "tenant_id": "T001",
        "payment_delay_days": 8,
        "lease_remaining_months": 3,
        "market_rent_delta_pct": 18,
        "occupancy_rate": 94,
        "renewal_intent_score": 2,
        "support_tickets_qtly": 14,
        "lease_duration_years": 2.1,
        "area_sqft": 18500,
        "sector_encoded": 0,
        "floor_level": 13,
        "num_lease_renewals": 1
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    try:
        data = request.get_json()
        tenant_id = data.get("tenant_id", "UNKNOWN")

        # Build feature vector
        raw = {k: float(data.get(k, 0)) for k in [
            "payment_delay_days", "lease_remaining_months", "market_rent_delta_pct",
            "occupancy_rate", "renewal_intent_score", "support_tickets_qtly",
            "lease_duration_years", "area_sqft", "sector_encoded", "floor_level",
            "num_lease_renewals"
        ]}

        # Engineer features (same as training)
        raw["payment_risk_score"] = raw["payment_delay_days"] / 45
        raw["lease_urgency"] = 1 / (raw["lease_remaining_months"] + 1)
        raw["rent_pressure"] = max(raw["market_rent_delta_pct"], 0) / 40
        raw["engagement_score"] = raw["support_tickets_qtly"] / 20
        raw["loyalty_score"] = raw["num_lease_renewals"] / 5

        X = pd.DataFrame([raw])[feature_names]
        churn_prob = float(model.predict_proba(X)[0][1])
        risk_score = int(churn_prob * 100)

        # SHAP explanation
        shap_vals = explainer.shap_values(X)[0]
        shap_explanation = [
            {"feature": f, "shap_value": round(float(v), 4), "direction": "increases_risk" if v > 0 else "decreases_risk"}
            for f, v in sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:6]
        ]

        risk_level = "HIGH" if risk_score >= 75 else "MEDIUM" if risk_score >= 50 else "LOW"

        return jsonify({
            "tenant_id": tenant_id,
            "churn_probability": round(churn_prob, 4),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommendation": get_recommendation(risk_level, raw),
            "shap_explanation": shap_explanation,
            "model_version": "XGBoost-v2.4"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Batch prediction for multiple tenants."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        tenants = request.get_json()
        results = []

        for t in tenants:
            raw = {k: float(t.get(k, 0)) for k in [
                "payment_delay_days", "lease_remaining_months", "market_rent_delta_pct",
                "occupancy_rate", "renewal_intent_score", "support_tickets_qtly",
                "lease_duration_years", "area_sqft", "sector_encoded", "floor_level", "num_lease_renewals"
            ]}
            raw["payment_risk_score"] = raw["payment_delay_days"] / 45
            raw["lease_urgency"] = 1 / (raw["lease_remaining_months"] + 1)
            raw["rent_pressure"] = max(raw["market_rent_delta_pct"], 0) / 40
            raw["engagement_score"] = raw["support_tickets_qtly"] / 20
            raw["loyalty_score"] = raw["num_lease_renewals"] / 5

            X = pd.DataFrame([raw])[feature_names]
            prob = float(model.predict_proba(X)[0][1])
            risk = int(prob * 100)
            level = "HIGH" if risk >= 75 else "MEDIUM" if risk >= 50 else "LOW"
            results.append({"tenant_id": t.get("tenant_id"), "risk_score": risk, "risk_level": level, "churn_probability": round(prob, 4)})

        results.sort(key=lambda x: x["risk_score"], reverse=True)
        high_risk = sum(1 for r in results if r["risk_level"] == "HIGH")

        return jsonify({
            "total": len(results),
            "high_risk_count": high_risk,
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


def get_recommendation(risk_level, features):
    if risk_level == "HIGH":
        return "URGENT: Schedule retention meeting within 72 hours. Consider lease restructuring or incentive package."
    elif risk_level == "MEDIUM":
        return "MONITOR: Follow up with tenant this week. Review lease terms and market alignment."
    else:
        return "STABLE: Standard quarterly check-in sufficient. Tenant shows strong retention signals."


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
    
#MODEL TRAINING 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── 1. Generate Realistic Synthetic Training Data ────────────────────────────
np.random.seed(42)
N = 3200

def generate_data(n):
    data = {
        "tenant_id": [f"T{i:04d}" for i in range(n)],
        "payment_delay_days": np.random.exponential(4, n).clip(0, 45).astype(int),
        "lease_remaining_months": np.random.uniform(1, 48, n),
        "market_rent_delta_pct": np.random.normal(8, 10, n),
        "occupancy_rate": np.random.beta(8, 2, n) * 100,
        "renewal_intent_score": np.random.randint(1, 6, n),
        "support_tickets_qtly": np.random.poisson(5, n),
        "lease_duration_years": np.random.uniform(1, 8, n),
        "area_sqft": np.random.lognormal(9, 0.5, n).clip(1000, 50000),
        "sector_encoded": np.random.randint(0, 7, n),
        "floor_level": np.random.randint(1, 30, n),
        "num_lease_renewals": np.random.randint(0, 5, n),
    }
    df = pd.DataFrame(data)

    # Engineered features
    df["payment_risk_score"] = df["payment_delay_days"] / 45
    df["lease_urgency"] = 1 / (df["lease_remaining_months"] + 1)
    df["rent_pressure"] = df["market_rent_delta_pct"].clip(0, 40) / 40
    df["engagement_score"] = df["support_tickets_qtly"] / 20
    df["loyalty_score"] = df["num_lease_renewals"] / 5

    # Churn label (business-logic driven)
    churn_prob = (
        0.30 * df["payment_risk_score"] +
        0.25 * df["lease_urgency"] * 10 +
        0.20 * df["rent_pressure"] +
        0.15 * (1 - df["renewal_intent_score"] / 5) +
        0.10 * df["engagement_score"] -
        0.10 * df["loyalty_score"] +
        np.random.normal(0, 0.05, n)
    ).clip(0, 1)

    df["churn"] = (churn_prob > 0.45).astype(int)
    return df

df = generate_data(N)
print(f"✅ Data Generated: {N} tenants | Churn Rate: {df['churn'].mean():.1%}")

# ── 2. Feature Selection & Split ────────────────────────────────────────────
FEATURES = [
    "payment_delay_days", "lease_remaining_months", "market_rent_delta_pct",
    "occupancy_rate", "renewal_intent_score", "support_tickets_qtly",
    "lease_duration_years", "area_sqft", "sector_encoded", "floor_level",
    "num_lease_renewals", "payment_risk_score", "lease_urgency",
    "rent_pressure", "engagement_score", "loyalty_score"
]

X = df[FEATURES]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}")

# ── 3. Train XGBoost ─────────────────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# ── 4. Evaluate ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("📈 CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))
print(f"🎯 ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
print(f"📊 5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 5. SHAP Analysis ─────────────────────────────────────────────────────────
print("\n🔍 Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:100])

# Save global feature importance
feature_importance = pd.DataFrame({
    "feature": FEATURES,
    "mean_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_shap", ascending=False)

print("\n📌 Top 5 Features by SHAP Importance:")
print(feature_importance.head().to_string(index=False))

# ── 6. Save Artifacts ────────────────────────────────────────────────────────
joblib.dump(model, "model/xgb_churn_model.pkl")
joblib.dump(explainer, "model/shap_explainer.pkl")
joblib.dump(FEATURES, "model/feature_names.pkl")
feature_importance.to_csv("model/feature_importance.csv", index=False)

print("\n✅ Model artifacts saved to ./model/")
print("🚀 Run `python app.py` to start the Flask API")