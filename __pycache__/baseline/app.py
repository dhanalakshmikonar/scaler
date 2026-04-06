from __future__ import annotations

import streamlit as st

from crisis_verify_env.predictor import predict_claim


st.set_page_config(
    page_title="CrisisVerify AI",
    page_icon="C",
    layout="wide",
)

st.markdown(
    """
    <style>
    .hero {
        padding: 1.5rem 1.75rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1d4ed8 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 16px;
        background: #f8fafc;
        border: 1px solid #dbeafe;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.35rem;">CrisisVerify AI</h1>
        <p style="font-size:1.05rem; margin-bottom:0;">
            Analyze crisis-related claims and estimate whether they look real, misleading, fake,
            or still unverified based on source cues, language patterns, and known misinformation styles.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.15, 1.0], gap="large")

with left:
    st.subheader("Analyze a Claim")
    claim = st.text_area(
        "Claim or Headline",
        placeholder="Example: Breaking: City hospital completely destroyed in overnight strike, all services collapsed.",
        height=180,
    )
    source_type = st.selectbox(
        "Source Type",
        [
            "viral social account",
            "telegram channel",
            "government channel",
            "news website",
            "anonymous leak channel",
            "private forward",
            "community forum",
        ],
    )
    include_media = st.checkbox("Claim includes image or video evidence", value=True)
    analyze = st.button("Analyze Claim", type="primary", use_container_width=True)

with right:
    st.subheader("What the app checks")
    st.markdown(
        """
        - source credibility patterns
        - whether the claim fits war/crisis context
        - timeline mismatch and media-reuse risk
        - emotional or propaganda-style language
        - similarity to known crisis misinformation scenarios
        """
    )
    st.info(
        "This version is fully offline. It does not search the web live, so it should be treated as a decision-support demo, not a final fact checker."
    )

if analyze:
    if not claim.strip():
        st.warning("Enter a claim or headline first.")
    else:
        result = predict_claim(claim=claim, source_type=source_type, include_media=include_media)

        verdict_label = result.verdict.value.replace("_", " ").title()
        if verdict_label == "Insufficient Evidence":
            verdict_label = "Unverified"

        verdict_color = {
            "Real": "#166534",
            "Misleading": "#b45309",
            "Fake": "#b91c1c",
            "Unverified": "#475569",
        }.get(verdict_label, "#475569")

        st.markdown("### Prediction")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:0.9rem;color:#475569;">Verdict</div>
                    <div style="font-size:1.5rem;font-weight:800;color:{verdict_color};">{verdict_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:0.9rem;color:#475569;">Confidence</div>
                    <div style="font-size:1.5rem;font-weight:800;color:#0f172a;">{int(result.confidence * 100)}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:0.9rem;color:#475569;">Risk Level</div>
                    <div style="font-size:1.1rem;font-weight:800;color:#0f172a;">{result.risk_level}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        detail_left, detail_right = st.columns([1.1, 0.9], gap="large")
        with detail_left:
            st.subheader("Why the AI predicted this")
            if result.scope_status == "out_of_scope":
                st.warning("This claim appears outside the app's intended crisis-misinformation scope.")
            st.write(result.explanation)
            st.caption(f"Matched investigation pattern: {result.matched_scenario_id} ({result.difficulty_hint.value})")

            st.subheader("Key signals")
            for signal in result.signals:
                st.markdown(f"- {signal}")

        with detail_right:
            st.subheader("Suggested verification checks")
            for check in result.suggested_checks:
                st.markdown(f"- {check}")

            st.subheader("Recommended use")
            st.markdown(
                """
                1. Use this for crisis-related claims, not general world facts.
                2. Treat the result as a warning or triage signal.
                3. Cross-check sensitive claims with trusted reporting.
                4. Be especially careful with deaths, leadership claims, and breaking war updates.
                """
            )
else:
    st.info("Enter a claim and click Analyze Claim to see the prediction.")
