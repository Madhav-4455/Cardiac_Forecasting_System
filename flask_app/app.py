from flask import (
    Flask, render_template, request,
    session, send_file, redirect, url_for
)
import pickle, joblib, pandas as pd, io, secrets, tempfile, os, random
from fpdf import FPDF
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
print(f"Generated Secret Key: {app.secret_key}")  # Print the key temporarily

# ----------------------------
# Load trained models (original absolute paths)
# ----------------------------
models = {
    'Logistic Regression': pickle.load(open(
        "D:/DA and DS Projects/Heart_Disease_Prediction/flask_app/models/logistic_regression_Model.pkl", "rb"
    )),
    'SVM': pickle.load(open(
        "D:/DA and DS Projects/Heart_Disease_Prediction/flask_app/models/svm_Model.pkl", "rb"
    )),
    'Decision Tree': pickle.load(open(
        "D:/DA and DS Projects/Heart_Disease_Prediction/flask_app/models/decision_tree_Model.pkl", "rb"
    )),
    'Random Forest': pickle.load(open(
        "D:/DA and DS Projects/Heart_Disease_Prediction/flask_app/models/random_forest_Model.pkl", "rb"
    )),
    'XGBoost': pickle.load(open(
        "D:/DA and DS Projects/Heart_Disease_Prediction/flask_app/models/xgboost_Model.pkl", "rb"
    ))
}

# ----------------------------
# Load preprocessors (original absolute paths)
# ----------------------------
scaler           = joblib.load(
    "D:/DA and DS Projects/Heart_Disease_Prediction/flask_app/models/scaler.pkl"
)
poly             = joblib.load(
    "D:/DA and DS Projects/Heart_Disease_Prediction/flask_app/models/poly.pkl"
)
expected_columns = joblib.load(
    "D:/DA and DS Projects/Heart_Disease_Prediction/flask_app/models/expected_columns.pkl"
)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # 1) Collect & cast inputs
        features = [
            'age','sex','chest pain type','resting bps','cholesterol',
            'fasting blood sugar','resting ecg','max heart rate','exercise angina',
            'oldpeak','ST slope'
        ]
        raw = { f: float(request.form[f]) for f in features }

        # 2) Build DataFrame + engineered features
        df = pd.DataFrame([raw])
        df['max heart rate_oldpeak'] = df['max heart rate'] * df['oldpeak']
        df['chest pain type_resting bps']     = df['chest pain type'] * df['resting bps']
        df['age_bins']        = pd.cut(
            df['age'], bins=[0,40,55,70,100], labels=[0,1,2,3]
        )
        df['cholesterol_bins']       = pd.cut(
            df['cholesterol'], bins=[0,200,240,600], labels=[0,1,2]
        )
        df = pd.get_dummies(df, columns=['age_bins','cholesterol_bins'])
        df = df.reindex(columns=expected_columns, fill_value=0)

        # 3) Polynomial & scaling
        X_poly   = poly.transform(df)
        X_scaled = scaler.transform(X_poly)

        # 4) Model predictions
        detailed = {}
        positives = 0
        for name, model in models.items():
            p = int(model.predict(X_scaled)[0])
            detailed[name] = (
                "High Chance of Heart Disease" if p == 1
                else "Low Chance of Heart Disease"
            )
            positives += p

        overall_pct  = round(positives / len(models) * 100)
        overall_text = f"{overall_pct}% chance that you have heart disease"

        # 5) Save for report, then re-render index.html with results
        session['report'] = {
            'inputs': raw,
            'overall_result': overall_text,
            'detailed': detailed,
            'overall_pct': overall_pct
        }

        return render_template(
            'index.html',
            show_results=True,
            overall_result=overall_text,
            detailed=detailed,
            overall_pct=overall_pct
        )

    # GET → blank form
    return render_template('index.html', show_results=False)


@app.route('/generate_report')
def generate_report():
    data = session.get('report')
    if not data:
        return redirect(url_for('home'))

    # Friendly labels
    feature_map = {
        'age':'Age','sex':'Gender','chest pain type':'Chest Pain Type',
        'resting bps':'Resting BP','cholesterol':'Cholesterol','fasting blood sugar':'Fasting BS',
        'resting ecg':'ECG','max heart rate':'Max HR','exercise angina':'Exercise Angina',
        'oldpeak':'ST Depression','ST Slope':'ST Slope'
    }

    # Initialize PDF
    pdf = FPDF()
    pdf.add_page()

    # Header with HeartGuard Branding
    pdf.set_fill_color(200, 30, 50)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 15, "HeartGuard Heart Disease Report", ln=True, align="C", fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # Subtitle
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(200, 30, 50)
    pdf.cell(0, 10, "Personalized Heart Health Analysis", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)

    # Inputs Section
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Your Inputs:", ln=True, fill=True)
    pdf.set_font("Arial", '', 12)
    pdf.ln(4)
    for k, v in data['inputs'].items():
        pdf.set_x(15)
        pdf.cell(90, 8, f"{feature_map.get(k, k)}:", border=1)
        pdf.cell(90, 8, str(v), border=1, ln=True)
    pdf.ln(8)

    # Overall Prediction Section
    pdf.set_fill_color(200, 30, 50)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Overall Prediction:", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.ln(4)
    pdf.set_x(15)
    pdf.multi_cell(0, 8, data['overall_result'], border=1)
    pdf.ln(8)

    # Detailed Model Results Section
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Model-by-Model Results:", ln=True, fill=True)
    pdf.set_font("Arial", '', 12)
    pdf.ln(4)
    for m, pred in data['detailed'].items():
        pdf.set_x(15)
        pdf.cell(90, 8, m, border=1, fill=True)
        pdf.cell(90, 8, pred, border=1, ln=True)
    pdf.ln(8)

    # Risk Trend Projection Chart
    pct = data['overall_pct']
    years = list(range(11))
    risk_points = [min(95, pct + year * 2.5 + random.random()) for year in years]

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(years, risk_points, marker='o')
    ax.set_title('10-Year Risk Trend Projection')
    ax.set_xlabel('Years from Now')
    ax.set_ylabel('Risk (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    # Write buffer to a temp PNG file for FPDF
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.write(buf.getvalue())
    tmp.close()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "10-Year Risk Trend Projection", ln=True)
    pdf.image(tmp.name, x=15, w=pdf.w - 30)
    pdf.ln(8)
    os.remove(tmp.name)

    # Footer Disclaimer
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(
        0, 6,
        "Disclaimer: This HeartGuard report is for informational purposes only. Consult a healthcare professional for medical advice."
    )
    pdf.set_text_color(0, 0, 0)

    # Generate PDF as bytes
    pdf_bytes = io.BytesIO(pdf.output(dest='S').encode('latin1'))
    return send_file(
        pdf_bytes,
        as_attachment=True,
        download_name="HeartGuard_Heart_Disease_Report.pdf",
        mimetype="application/pdf"
    )


if __name__ == '__main__':
    app.run(debug=True)

  #                                   python flask_app\app.py