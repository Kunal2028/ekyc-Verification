import requests
import streamlit as st
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="eKYC Face Verification",
    page_icon="🔐",
    layout="wide",
)

st.title("🔐 eKYC Face Verification System")
st.markdown(
    "Upload your **Government ID** and a **Live Selfie** to verify your identity."
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")

    threshold = st.slider(
        "Similarity Threshold",
        min_value=0.10,
        max_value=0.80,
        value=0.35,
        step=0.01,
    )

    st.markdown("---")

    if st.button("Check API Health"):
        try:
            response = requests.get(f"{API_URL}/health")

            if response.status_code == 200:
                st.success("API is running ✅")
            else:
                st.error("API returned an error.")

        except:
            st.error("Cannot connect to API.")


# -----------------------------
# Upload Section
# -----------------------------
col1, col2 = st.columns(2)

with col1:

    st.subheader("📄 Government ID")

    id_image = st.file_uploader(
        "Choose ID Image",
        type=["jpg", "jpeg", "png"],
        key="id",
    )

    if id_image:
        st.image(
            Image.open(id_image),
            caption="Uploaded ID",
            use_container_width=True,
        )

with col2:

    st.subheader("🤳 Live Selfie")

    live_image = st.file_uploader(
        "Choose Live Image",
        type=["jpg", "jpeg", "png"],
        key="live",
    )

    if live_image:
        st.image(
            Image.open(live_image),
            caption="Uploaded Selfie",
            use_container_width=True,
        )

st.divider()

# -----------------------------
# Verify Button
# -----------------------------
verify = st.button(
    "🔍 Verify Identity",
    type="primary",
    use_container_width=True,
)

# -----------------------------
# Verification
# -----------------------------
if verify:

    if id_image is None:
        st.error("Please upload an ID image.")
        st.stop()

    if live_image is None:
        st.error("Please upload a live image.")
        st.stop()

    files = {
        "id_image": (
            id_image.name,
            id_image.getvalue(),
            id_image.type,
        ),
        "live_image": (
            live_image.name,
            live_image.getvalue(),
            live_image.type,
        ),
    }

    params = {
        "threshold": threshold,
    }

    with st.spinner("Verifying identity..."):

        try:
            response = requests.post(
                f"{API_URL}/match-faces",
                files=files,
                params=params,
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to FastAPI backend.\n\n"
                "Start it using:\n"
                "uvicorn api:app --reload"
            )
            st.stop()

    if response.status_code != 200:

        try:
            error = response.json()["detail"]
        except:
            error = "Unknown error"

        st.error(error)
        st.stop()

    data = response.json()

    st.divider()

    st.header("Verification Result")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Similarity Score",
            f"{data['similarity_score']:.4f}",
        )

    with c2:
        st.metric(
            "Threshold",
            f"{data['threshold']:.2f}",
        )

    with c3:
        if data["matched"]:
            st.metric(
                "Status",
                "MATCHED ✅",
            )
        else:
            st.metric(
                "Status",
                "NOT MATCHED ❌",
            )

    st.divider()

    if data["matched"]:

        st.success("🎉 Identity Verified Successfully!")
        st.balloons()

    else:

        st.error("❌ Identity Verification Failed.")

    st.write("### Message")
    st.info(data["message"])

    st.progress(
        min(data["similarity_score"], 1.0)
    )

    if data["similarity_score"] >= threshold:
        st.success(
            f"Similarity ({data['similarity_score']:.4f}) is above the threshold ({threshold:.2f})."
        )
    else:
        st.warning(
            f"Similarity ({data['similarity_score']:.4f}) is below the threshold ({threshold:.2f})."
        )
