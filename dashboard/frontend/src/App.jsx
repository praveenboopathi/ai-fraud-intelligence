import { useState } from "react";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [amount, setAmount] = useState(1000);
  const [time, setTime] = useState(3600);

  const [supervised, setSupervised] = useState(null);
  const [unsupervised, setUnsupervised] = useState(null);

  const [loadingSupervised, setLoadingSupervised] = useState(false);
  const [loadingUnsupervised, setLoadingUnsupervised] = useState(false);
  const [error, setError] = useState("");

  const predictSupervised = async () => {
    setLoadingSupervised(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/predict/supervised`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          Amount: Number(amount),
          Time: Number(time),
        }),
      });

      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }

      const data = await response.json();
      setSupervised(data);
    } catch (err) {
      setError(
        "Could not connect to the FastAPI backend. Make sure the backend is running on port 8000."
      );
    } finally {
      setLoadingSupervised(false);
    }
  };

  const predictUnsupervised = async () => {
    setLoadingUnsupervised(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/predict/unsupervised`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          Amount: Number(amount),
          Time: Number(time),
        }),
      });

      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }

      const data = await response.json();
      setUnsupervised(data);
    } catch (err) {
      setError(
        "Could not connect to the FastAPI backend. Make sure the backend is running on port 8000."
      );
    } finally {
      setLoadingUnsupervised(false);
    }
  };

  const runAnalysis = async () => {
    await Promise.all([predictSupervised(), predictUnsupervised()]);
  };

  const isFraud = supervised?.prediction === 1;
  const isAnomaly = unsupervised?.anomaly === 1;
  const riskExplanation =
  supervised?.risk_level === "HIGH"
    ? "High risk: the model indicates strong fraud characteristics."
    : supervised?.risk_level === "MEDIUM"
      ? "Medium risk: the transaction shows some suspicious characteristics."
      : "Low risk: the transaction shows fewer suspicious characteristics.";

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <div>
          <div style={styles.badge}>AI FRAUD INTELLIGENCE</div>
          <h1 style={styles.title}>Fraud Detection Dashboard</h1>
          <p style={styles.subtitle}>
            Real-time transaction analysis using supervised and unsupervised
            machine learning.
          </p>
        </div>

        <div style={styles.status}>
          <span style={styles.statusDot}></span>
          API ONLINE
        </div>
      </header>

      <main style={styles.container}>
        <section style={styles.inputCard}>
          <div style={styles.sectionTitle}>Transaction Analysis</div>
          <p style={styles.sectionSubtitle}>
            Enter transaction details and run both fraud detection models.
          </p>

          <div style={styles.formGrid}>
            <div>
              <label style={styles.label}>Transaction Amount</label>
              <input
                style={styles.input}
                type="number"
                min="0"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
              />
            </div>

            <div>
              <label style={styles.label}>Transaction Time</label>
              <input
                style={styles.input}
                type="number"
                min="0"
                value={time}
                onChange={(e) => setTime(e.target.value)}
              />
            </div>
          </div>

          <button style={styles.primaryButton} onClick={runAnalysis}>
            {loadingSupervised || loadingUnsupervised
              ? "Analyzing Transaction..."
              : "Analyze Transaction"}
          </button>

          {error && <div style={styles.error}>{error}</div>}
        </section>

        <section style={styles.grid}>
          <div style={styles.resultCard}>
            <div style={styles.cardHeader}>
              <span style={styles.icon}>◉</span>
              <div>
                <h2 style={styles.cardTitle}>Supervised Detection</h2>
                <p style={styles.cardSubtitle}>Classification model</p>
              </div>
            </div>

            {supervised ? (
              <>
                <div
                  style={{
                    ...styles.resultBanner,
                    ...(isFraud ? styles.danger : styles.safe),
                  }}
                >
                  <strong>
                    {isFraud ? "FRAUD DETECTED" : "TRANSACTION NORMAL"}
                  </strong>
                  <span>
                    {supervised.fraud_probability !== undefined
                      ? `${(
                          supervised.fraud_probability * 100
                        ).toFixed(2)}% probability`
                      : `Prediction: ${supervised.prediction}`}
                  </span>
                </div>

                <div style={styles.metricsGrid}>
                  <Metric
                    label="Risk Level"
                    value={supervised.risk_level ?? "N/A"}
                  />
               
                  <Metric
                    label="Precision"
                    value={formatMetric(supervised.precision)}
                  />
                  <Metric
                    label="Recall"
                    value={formatMetric(supervised.recall)}
                  />
                  <Metric label="F1 Score" value={formatMetric(supervised.f1)} />
                  <Metric
                    label="ROC-AUC"
                    value={formatMetric(supervised.roc_auc)}
                  />
                  <Metric
                    label="PR-AUC"
                    value={formatMetric(supervised.pr_auc)}
                  />
                </div>
              </>
            ) : (
              <EmptyState text="Run the analysis to get a supervised prediction." />
            )}
          </div>

          <div style={styles.resultCard}>
            <div style={styles.cardHeader}>
              <span style={styles.icon}>◇</span>
              <div>
                <h2 style={styles.cardTitle}>Anomaly Detection</h2>
                <p style={styles.cardSubtitle}>Isolation Forest model</p>
              </div>
            </div>

            {unsupervised ? (
              <div
                style={{
                  ...styles.anomalyBox,
                  ...(isAnomaly ? styles.anomaly : styles.normal),
                }}
              >
                <div style={styles.anomalyNumber}>
                  {isAnomaly ? "1" : "0"}
                </div>

                <div>
                  <strong style={styles.anomalyTitle}>
                    {isAnomaly ? "ANOMALY DETECTED" : "NO ANOMALY"}
                  </strong>
                  <p style={styles.anomalyText}>
                    {isAnomaly
                      ? "The transaction has unusual characteristics."
                      : "The transaction appears normal according to the anomaly model."}
                  </p>
                </div>
              </div>
            ) : (
              <EmptyState text="Run the analysis to check for anomalies." />
            )}
          </div>
        </section>

        <section style={styles.overviewCard}>
          <div style={styles.sectionTitle}>Model Overview</div>

          <div style={styles.overviewGrid}>
            <Overview
              title="Supervised Model"
              text="Uses labeled transaction data to classify transactions as fraudulent or legitimate."
            />

            <Overview
              title="Unsupervised Model"
              text="Uses Isolation Forest to identify transactions with unusual patterns."
            />

            <Overview
              title="Feature Engineering"
              text="Transaction amount and time are transformed using the project's feature engineering pipeline."
            />

            <Overview
              title="API Architecture"
              text="React communicates with the FastAPI backend through REST API endpoints."
            />
          </div>
        </section>
      </main>

      <footer style={styles.footer}>
        AI Fraud Intelligence • FastAPI + React + Machine Learning
      </footer>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div style={styles.metric}>
      <span style={styles.metricLabel}>{label}</span>
      <strong style={styles.metricValue}>{value}</strong>
    </div>
  );
}

function Overview({ title, text }) {
  return (
    <div style={styles.overviewItem}>
      <h3 style={styles.overviewTitle}>{title}</h3>
      <p style={styles.overviewText}>{text}</p>
    </div>
  );
}

function EmptyState({ text }) {
  return <div style={styles.empty}>{text}</div>;
}

function formatMetric(value) {
  if (value === undefined || value === null) {
    return "N/A";
  }

  return Number(value).toFixed(4);
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "#0b1120",
    color: "#e5e7eb",
    fontFamily:
      "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
  },

  header: {
    padding: "36px 7%",
    borderBottom: "1px solid #1f2937",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: "30px",
  },

  badge: {
    display: "inline-block",
    fontSize: "12px",
    fontWeight: "700",
    letterSpacing: "2px",
    color: "#60a5fa",
    marginBottom: "10px",
  },

  title: {
    margin: 0,
    fontSize: "38px",
    lineHeight: 1.15,
    color: "#f8fafc",
  },

  subtitle: {
    margin: "12px 0 0",
    color: "#94a3b8",
    maxWidth: "700px",
    fontSize: "16px",
  },

  status: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    padding: "10px 16px",
    borderRadius: "999px",
    border: "1px solid #14532d",
    background: "#052e16",
    color: "#86efac",
    fontSize: "12px",
    fontWeight: "700",
    whiteSpace: "nowrap",
  },

  statusDot: {
    width: "8px",
    height: "8px",
    borderRadius: "50%",
    background: "#22c55e",
  },

  container: {
    width: "86%",
    maxWidth: "1300px",
    margin: "0 auto",
    padding: "40px 0",
  },

  inputCard: {
    background: "#111827",
    border: "1px solid #1f2937",
    borderRadius: "18px",
    padding: "28px",
    marginBottom: "24px",
  },

  sectionTitle: {
    fontSize: "22px",
    fontWeight: "700",
    color: "#f8fafc",
  },

  sectionSubtitle: {
    color: "#94a3b8",
    marginTop: "7px",
  },

  formGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "18px",
    marginTop: "25px",
  },

  label: {
    display: "block",
    fontSize: "13px",
    fontWeight: "600",
    color: "#cbd5e1",
    marginBottom: "8px",
  },

  input: {
    width: "100%",
    boxSizing: "border-box",
    padding: "13px 14px",
    borderRadius: "10px",
    border: "1px solid #334155",
    background: "#020617",
    color: "#f8fafc",
    fontSize: "16px",
    outline: "none",
  },

  primaryButton: {
    marginTop: "22px",
    width: "100%",
    padding: "14px",
    border: "none",
    borderRadius: "10px",
    background: "#2563eb",
    color: "white",
    fontSize: "15px",
    fontWeight: "700",
    cursor: "pointer",
  },

  error: {
    marginTop: "15px",
    padding: "12px",
    borderRadius: "8px",
    background: "#450a0a",
    border: "1px solid #7f1d1d",
    color: "#fca5a5",
  },

  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "24px",
  },

  resultCard: {
    background: "#111827",
    border: "1px solid #1f2937",
    borderRadius: "18px",
    padding: "26px",
    minHeight: "320px",
  },

  cardHeader: {
    display: "flex",
    alignItems: "center",
    gap: "14px",
    marginBottom: "24px",
  },

  icon: {
    width: "42px",
    height: "42px",
    borderRadius: "10px",
    display: "grid",
    placeItems: "center",
    background: "#172554",
    color: "#60a5fa",
    fontSize: "20px",
  },

  cardTitle: {
    margin: 0,
    fontSize: "19px",
    color: "#f8fafc",
  },

  cardSubtitle: {
    margin: "4px 0 0",
    color: "#64748b",
    fontSize: "13px",
  },

  resultBanner: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: "15px",
    padding: "16px",
    borderRadius: "12px",
    marginBottom: "18px",
  },

  safe: {
    background: "#052e16",
    border: "1px solid #166534",
    color: "#86efac",
  },

  danger: {
    background: "#450a0a",
    border: "1px solid #991b1b",
    color: "#fca5a5",
  },

  metricsGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: "10px",
  },
  explanation: {
  marginTop: "14px",
  color: "#94a3b8",
  fontSize: "14px",
  lineHeight: 1.5,
  },

  metric: {
    background: "#0f172a",
    border: "1px solid #1e293b",
    borderRadius: "10px",
    padding: "13px",
  },

  metricLabel: {
    display: "block",
    color: "#64748b",
    fontSize: "11px",
    marginBottom: "6px",
  },

  metricValue: {
    color: "#e2e8f0",
    fontSize: "15px",
  },

  anomalyBox: {
    display: "flex",
    alignItems: "center",
    gap: "20px",
    padding: "25px",
    borderRadius: "14px",
  },

  normal: {
    background: "#052e16",
    border: "1px solid #166534",
  },

  anomaly: {
    background: "#450a0a",
    border: "1px solid #991b1b",
  },

  anomalyNumber: {
    fontSize: "54px",
    fontWeight: "800",
    lineHeight: 1,
  },

  anomalyTitle: {
    fontSize: "17px",
  },

  anomalyText: {
    color: "#94a3b8",
    lineHeight: 1.5,
    marginBottom: 0,
  },

  empty: {
    display: "grid",
    placeItems: "center",
    minHeight: "220px",
    textAlign: "center",
    color: "#64748b",
    border: "1px dashed #334155",
    borderRadius: "12px",
    padding: "20px",
  },

  overviewCard: {
    marginTop: "24px",
    background: "#111827",
    border: "1px solid #1f2937",
    borderRadius: "18px",
    padding: "28px",
  },

  overviewGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(2, 1fr)",
    gap: "18px",
    marginTop: "22px",
  },

  overviewItem: {
    padding: "20px",
    background: "#0f172a",
    borderRadius: "12px",
    border: "1px solid #1e293b",
  },

  overviewTitle: {
    margin: "0 0 8px",
    color: "#60a5fa",
    fontSize: "16px",
  },

  overviewText: {
    margin: 0,
    color: "#94a3b8",
    lineHeight: 1.6,
    fontSize: "14px",
  },

  footer: {
    borderTop: "1px solid #1f2937",
    padding: "24px",
    textAlign: "center",
    color: "#64748b",
    fontSize: "13px",
  },
};

export default App;
