(() => {
  const form = document.getElementById("riskForm");
  const submitBtn = document.getElementById("submitBtn");
  const errorNote = document.getElementById("errorNote");
  const verdict = document.getElementById("verdict");

  const gaugeFill = document.getElementById("gaugeFill");
  const gaugeThreshold = document.getElementById("gaugeThreshold");
  const probNumber = document.getElementById("probNumber");
  const stampBadge = document.getElementById("stampBadge");
  const stampText = document.getElementById("stampText");

  const factProb = document.getElementById("factProb");
  const factThreshold = document.getElementById("factThreshold");
  const factResult = document.getElementById("factResult");

  const apiDot = document.getElementById("apiDot");
  const apiStatusText = document.getElementById("apiStatusText");

  const GAUGE_CIRCUMFERENCE = 540.35; // 2 * PI * 86, matches CSS

  // ---------- Service status check ----------
  fetch("/openapi.json", { method: "GET" })
    .then((res) => {
      if (res.ok) {
        apiDot.classList.add("ok");
        apiStatusText.textContent = "service ready";
      } else {
        throw new Error("bad status");
      }
    })
    .catch(() => {
      apiDot.classList.add("down");
      apiStatusText.textContent = "service unreachable";
    });

  // ---------- Helpers ----------
  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitBtn.classList.toggle("loading", isLoading);
    submitBtn.querySelector(".btn-label").textContent = isLoading
      ? "Analyzing profile…"
      : "Assess attrition risk";
  }

  function showError(message) {
    errorNote.textContent = message;
    errorNote.hidden = false;
  }

  function clearError() {
    errorNote.hidden = true;
    errorNote.textContent = "";
  }

  function animateNumber(el, from, to, duration) {
    const start = performance.now();
    function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const value = from + (to - from) * eased;
      el.textContent = value.toFixed(1);
      if (t < 1) requestAnimationFrame(tick);
      else el.textContent = to.toFixed(1);
    }
    requestAnimationFrame(tick);
  }

  function renderVerdict(data) {
    const probabilityPct = data.default_probability * 100;
    const thresholdPct = data.threshold * 100;
    const isHighRisk = data.default_prediction === 1;

    verdict.hidden = false;
    verdict.scrollIntoView({ behavior: "smooth", block: "nearest" });

    // Gauge fill
    const offset = GAUGE_CIRCUMFERENCE * (1 - probabilityPct / 100);
    gaugeFill.style.stroke = isHighRisk ? "var(--risk-red)" : "var(--risk-green)";
    requestAnimationFrame(() => {
      gaugeFill.style.strokeDashoffset = offset;
    });

    // Threshold tick
    gaugeThreshold.style.transform = `rotate(${thresholdPct * 3.6}deg)`;

    // Number readout
    animateNumber(probNumber, 0, probabilityPct, 1000);

    // Stamp
    stampBadge.classList.remove("stamp--in", "risk-high");
    void stampBadge.offsetWidth; // restart animation
    if (isHighRisk) {
      stampBadge.classList.add("risk-high");
      stampText.textContent = "ATTRITION";
    } else {
      stampText.textContent = "RETENTION";
    }
    requestAnimationFrame(() => stampBadge.classList.add("stamp--in"));

    // Ledger facts
    factProb.textContent = `${probabilityPct.toFixed(1)}%`;
    factThreshold.textContent = `${thresholdPct.toFixed(1)}%`;
    factResult.textContent = data.Result;
  }

  // ---------- Submit ----------
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();
    setLoading(true);

    const payload = {
      Age: parseInt(document.getElementById("Age").value, 10),
      BusinessTravel: document.getElementById("BusinessTravel").value,
      DailyRate: parseInt(document.getElementById("DailyRate").value, 10),
      Department: document.getElementById("Department").value,
      DistanceFromHome: parseInt(document.getElementById("DistanceFromHome").value, 10),
      Education: parseInt(document.getElementById("Education").value, 10),
      EducationField: document.getElementById("EducationField").value,
      EnvironmentSatisfaction: parseInt(document.getElementById("EnvironmentSatisfaction").value, 10),
      Gender: document.getElementById("Gender").value,
      HourlyRate: parseInt(document.getElementById("HourlyRate").value, 10),
      JobInvolvement: parseInt(document.getElementById("JobInvolvement").value, 10),
      JobLevel: parseInt(document.getElementById("JobLevel").value, 10),
      JobRole: document.getElementById("JobRole").value,
      JobSatisfaction: parseInt(document.getElementById("JobSatisfaction").value, 10),
      MaritalStatus: document.getElementById("MaritalStatus").value,
      MonthlyIncome: parseFloat(document.getElementById("MonthlyIncome").value),
      MonthlyRate: parseInt(document.getElementById("MonthlyRate").value, 10),
      NumCompaniesWorked: parseInt(document.getElementById("NumCompaniesWorked").value, 10),
      OverTime: document.getElementById("OverTime").value,
      PercentSalaryHike: parseInt(document.getElementById("PercentSalaryHike").value, 10),
      PerformanceRating: parseInt(document.getElementById("PerformanceRating").value, 10),
      RelationshipSatisfaction: parseInt(document.getElementById("RelationshipSatisfaction").value, 10),
      StockOptionLevel: parseInt(document.getElementById("StockOptionLevel").value, 10),
      TotalWorkingYears: parseFloat(document.getElementById("TotalWorkingYears").value),
      TrainingTimesLastYear: parseInt(document.getElementById("TrainingTimesLastYear").value, 10),
      WorkLifeBalance: parseInt(document.getElementById("WorkLifeBalance").value, 10),
      YearsAtCompany: parseFloat(document.getElementById("YearsAtCompany").value),
      YearsInCurrentRole: parseInt(document.getElementById("YearsInCurrentRole").value, 10),
      YearsSinceLastPromotion: parseFloat(document.getElementById("YearsSinceLastPromotion").value),
      YearsWithCurrManager: parseInt(document.getElementById("YearsWithCurrManager").value, 10)
    };

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => null);
        const detail = body && body.detail ? JSON.stringify(body.detail) : `HTTP ${res.status}`;
        throw new Error(detail);
      }

      const data = await res.json();
      renderVerdict(data);
    } catch (err) {
      showError(`Could not reach the ledger. ${err.message || "Check the service is running."}`);
    } finally {
      setLoading(false);
    }
  });
})();
