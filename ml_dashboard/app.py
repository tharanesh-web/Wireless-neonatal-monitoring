from flask import Flask, render_template, request
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
model = joblib.load("model.pkl")
last_df = pd.DataFrame()  # Global to reuse for full data page

@app.route("/", methods=["GET", "POST"])
def index():
    global last_df
    predictions = None
    plot_paths = {}
    high_risk_detected = False
    mitigation_text = ""
    dataset_head = None

    if request.method == "POST":
        file = request.files["csv_file"]
        if file:
            df = pd.read_csv(file)
            df.drop(columns=["Timestamp"], inplace=True, errors='ignore')

            dataset_head = df.head().to_html(classes="table table-striped", index=False)

            try:
                # Preprocess the dataset
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]

                # Encode string labels to numeric using LabelEncoder
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)

                # Apply SMOTE to handle class imbalance
                sm = SMOTE(random_state=42)
                X_res, y_res = sm.fit_resample(X, y_encoded)

                # Initialize the RandomForestClassifier and train the model
                model.fit(X_res, y_res)

                # Predict using the trained model
                y_pred = model.predict(X)

                # Convert predictions to the same type as 'Risk Category'
                y_pred = label_encoder.inverse_transform(y_pred)

                # Ensure the "Prediction" column is the same type as 'Risk Category' (string)
                df["Prediction"] = y_pred.astype(str)
                last_df = df.copy()

                # Prediction Summary Table
                pred_counts = df["Prediction"].value_counts().reset_index()
                pred_counts.columns = ["Risk Category", "Count"]
                predictions = pred_counts.to_html(classes="table table-bordered", index=False)

                # Plot: Distribution of Risk Categories
                fig1, ax1 = plt.subplots()
                pred_counts.set_index("Risk Category").plot(kind="bar", ax=ax1, legend=False, color="orange")
                plt.title("Prediction Distribution")
                plt.ylabel("Count")
                plt.xticks(rotation=0)
                plt.tight_layout()
                dist_path = "static/plot.png"
                fig1.savefig(dist_path)
                plot_paths["dist"] = dist_path

                # High-risk alert
                if "High Risk" in pred_counts["Risk Category"].values:
                    high_risk_detected = True
                    mitigation_text = """
                    <ul>
                        <li>👶 Consult neonatal specialist immediately</li>
                        <li>📈 Monitor heart rate, SpO₂, RR continuously</li>
                        <li>💊 Ensure emergency response kit ready</li>
                        <li>📞 Inform pediatrician & update parents</li>
                    </ul>
                    """

                # Confusion Matrix + Classification Report
                y_true = df["Risk Category"].astype(str)  # Ensure this is string type
                cm = confusion_matrix(y_true, df["Prediction"], labels=np.unique(y_pred))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_pred))
                fig2, ax2 = plt.subplots()
                disp.plot(ax=ax2, cmap='Reds')
                plt.title("Confusion Matrix")
                plt.tight_layout()
                cm_path = "static/confusion.png"
                fig2.savefig(cm_path)
                plot_paths["confusion"] = cm_path

                # F1 Score Plot (manually from report)
                report_dict = classification_report(y_true, df["Prediction"], output_dict=True)
                f1_scores = {k: v["f1-score"] for k, v in report_dict.items() if isinstance(v, dict)}
                fig3, ax3 = plt.subplots()
                ax3.barh(list(f1_scores.keys()), list(f1_scores.values()), color="skyblue")
                ax3.set_title("F1 Score per Class")
                plt.tight_layout()
                f1_path = "static/f1score.png"
                fig3.savefig(f1_path)
                plot_paths["f1score"] = f1_path

            except Exception as e:
                predictions = f"<p class='text-danger'>Error: {str(e)}</p>"

    return render_template("index.html", predictions=predictions, dataset_head=dataset_head,
                           plot_paths=plot_paths, high_risk=high_risk_detected, mitigation=mitigation_text)

@app.route("/full-data")
def full_data():
    global last_df
    if not last_df.empty:
        html_table = last_df.to_html(classes="table table-hover", index=False)
    else:
        html_table = "<p>No data uploaded yet.</p>"
    return render_template("full_data.html", full_table=html_table)

if __name__ == "__main__":
    app.run(debug=True)
