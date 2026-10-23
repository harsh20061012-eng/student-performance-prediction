import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Student Performance AI", page_icon="🎓", layout="wide")

FEATURES = [
    "study_hours",
    "attendance",
    "previous_marks",
    "sleep_hours",
    "assignment_completion"
]
TARGET = "exam_score"


@st.cache_data
def create_data(n=600):
    rng = np.random.default_rng(42)

    study = rng.uniform(0.5, 10, n)
    attendance = rng.uniform(45, 100, n)
    previous = rng.uniform(30, 100, n)
    sleep = rng.uniform(4, 10, n)
    assignments = rng.uniform(30, 100, n)

    score = (
        4.2 * study
        + 0.20 * attendance
        + 0.28 * previous
        + 0.75 * sleep
        + 0.12 * assignments
        + rng.normal(0, 5, n)
        - 12
    )
    score = np.clip(score, 0, 100)

    return pd.DataFrame({
        "student_id": np.arange(1, n + 1),
        "study_hours": study.round(1),
        "attendance": attendance.round(1),
        "previous_marks": previous.round(1),
        "sleep_hours": sleep.round(1),
        "assignment_completion": assignments.round(1),
        "exam_score": score.round(1)
    })


st.title("🎓 Student Performance Analytics & Prediction")
st.caption("Machine Learning dashboard for analyzing and predicting student exam scores.")

with st.sidebar:
    st.header("⚙️ Dashboard Settings")
    uploaded = st.file_uploader("Upload student CSV (optional)", type=["csv"])
    model_name = st.selectbox(
        "Choose prediction model",
        ["Random Forest", "Linear Regression", "Gradient Boosting"]
    )

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
        if missing:
            st.error("Your CSV is missing these columns: " + ", ".join(missing))
            st.stop()

        for c in FEATURES + [TARGET]:
            data[c] = pd.to_numeric(data[c], errors="coerce")

        data = data.dropna(subset=[TARGET])
        if len(data) < 10:
            st.error("Please upload at least 10 rows with valid exam scores.")
            st.stop()

        for c in FEATURES:
            data[c] = data[c].fillna(data[c].median())

        data = data.dropna(subset=FEATURES)
        if len(data) < 10:
            st.error("Not enough valid rows to train the model.")
            st.stop()

    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    data = create_data()

X = data[FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=150, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

for model in models.values():
    model.fit(X_train, y_train)

model = models[model_name]
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "🔮 Predict Score", "📈 Analytics", "🧪 Model Evaluation"
])

with tab1:
    st.subheader("Student Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", f"{len(data):,}")
    c2.metric("Average Exam Score", f"{data[TARGET].mean():.1f}/100")
    c3.metric("Average Attendance", f"{data['attendance'].mean():.1f}%")
    c4.metric("Average Study Hours", f"{data['study_hours'].mean():.1f} hrs/day")

    st.plotly_chart(
        px.histogram(data, x=TARGET, nbins=20, title="Distribution of Exam Scores"),
        width='stretch'
    )
    st.dataframe(data.head(20), width='stretch')

with tab2:
    st.subheader("Predict a Student's Exam Score")
    st.write("Adjust the values below, then click **Predict Score**.")

    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.slider("Study hours per day", 0.0, 12.0, 4.0, 0.5)
        attendance = st.slider("Attendance (%)", 0.0, 100.0, 80.0, 1.0)
        previous_marks = st.slider("Previous exam marks", 0.0, 100.0, 70.0, 1.0)

    with col2:
        sleep_hours = st.slider("Sleep hours per night", 0.0, 12.0, 7.0, 0.5)
        assignment_completion = st.slider("Assignment completion (%)", 0.0, 100.0, 80.0, 1.0)

    if st.button("🔮 Predict Score", type="primary"):
        student = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_marks": previous_marks,
            "sleep_hours": sleep_hours,
            "assignment_completion": assignment_completion
        }])

        score = float(model.predict(student)[0])
        score = np.clip(score, 0, 100)
        st.success(f"Predicted Exam Score: {score:.1f} / 100")

        if score >= 80:
            st.write("Performance estimate: Excellent 🌟")
        elif score >= 60:
            st.write("Performance estimate: Good 👍")
        elif score >= 40:
            st.write("Performance estimate: Average 📘")
        else:
            st.write("Performance estimate: Needs Improvement 💡")

        st.caption("This is a model estimate, not a guaranteed exam result.")

with tab3:
    st.subheader("Explore Student Data")

    col1, col2 = st.columns(2)
    with col1:
        feature = st.selectbox("Choose a feature", FEATURES)
        st.plotly_chart(
            px.scatter(
                data, x=feature, y=TARGET,
                title=f"{feature.replace('_', ' ').title()} vs Exam Score",
                trendline="ols"
            ),
            width='stretch'
        )

    with col2:
        corr = data[FEATURES + [TARGET]].corr(numeric_only=True)
        st.plotly_chart(
            px.imshow(
                corr, text_auto=".2f", aspect="auto",
                title="Feature Correlation Heatmap",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1
            ),
            width='stretch'
        )

with tab4:
    st.subheader(f"Model Evaluation: {model_name}")

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.2f}")
    c2.metric("RMSE", f"{rmse:.2f}")
    c3.metric("R² Score", f"{r2:.3f}")

    st.caption("Lower MAE/RMSE is generally better. R² closer to 1 indicates a better fit.")

    comparison = []
    for name, fitted_model in models.items():
        p = fitted_model.predict(X_test)
        comparison.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, p),
            "RMSE": np.sqrt(mean_squared_error(y_test, p)),
            "R²": r2_score(y_test, p)
        })

    comparison_df = pd.DataFrame(comparison).sort_values("RMSE")
    st.dataframe(comparison_df.round(3), width='stretch')

    st.plotly_chart(
        px.bar(
            comparison_df, x="Model", y="RMSE",
            title="Compare Models by RMSE"
        ),
        width='stretch'
    )

st.sidebar.markdown("---")
st.sidebar.caption("Student Performance AI • Educational demo")
