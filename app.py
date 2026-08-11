```python
import requests
import streamlit as st
from PIL import Image

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="eKYC Verification",
    page_icon="🔐",
    layout="wide",
)

st.title("🔐 eKYC Verification System")
st.markdown(
    "Upload your **Government ID** and a **Live Video** "
    "to perform liveness detection and face verification."
)

with st.sidebar:
    st.header("Settings")

    threshold = st.slider(
        "Face Similarity Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.8,
        step=0.01,
    )

    st.markdown("---")

    if st.button("Check API Health"):
        try:
            response = requests.get(
                f"{API_URL}/health",
                timeout=10,
            )

            if response.status_code == 200:
                health_data = response.json()

                st.success("API is running ✅")

                st.write(
                    "Liveness module:",
                    health_data.get(
                        "liveness_module",
                        "Unknown",
                    ),
                )

                st.write(
                    "Face matching module:",
                    health_data.get(
                        "face_matching_module",
                        "Unknown",
                    ),
                )
            else:
                st.error("API returned an error.")

        except requests.exceptions.RequestException:
            st.error("Cannot connect to API.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Government ID")

    id_image = st.file_uploader(
        "Choose ID Image",
        type=["jpg", "jpeg", "png"],
        key="id_image",
    )

    if id_image is not None:
        try:
            image = Image.open(id_image)

            st.image(
                image,
                caption="Uploaded Government ID",
                use_container_width=True,
            )

            id_image.seek(0)

        except Exception:
            st.error("Unable to display the uploaded ID image.")

with col2:
    st.subheader("🎥 Live Video")

    live_video = st.file_uploader(
        "Choose Live Video",
        type=["mp4", "webm", "mov", "avi"],
        key="live_video",
    )

    if live_video is not None:
        st.video(live_video)

st.divider()

verify = st.button(
    "🔍 Verify Identity",
    type="primary",
    use_container_width=True,
)

if verify:
    if id_image is None:
        st.error("Please upload an ID image.")
        st.stop()

    if live_video is None:
        st.error("Please upload a live video.")
        st.stop()

    id_image.seek(0)
    live_video.seek(0)

    files = {
        "id_image": (
            id_image.name,
            id_image.getvalue(),
            id_image.type or "image/jpeg",
        ),
        "live_video": (
            live_video.name,
            live_video.getvalue(),
            live_video.type or "video/mp4",
        ),
    }

    params = {
        "face_threshold": threshold,
    }

    with st.spinner(
        "Checking liveness and verifying identity..."
    ):
        try:
            response = requests.post(
                f"{API_URL}/verify-ekyc",
                files=files,
                params=params,
                timeout=180,
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the FastAPI backend.\n\n"
                "Start it using:\n\n"
                "`uvicorn api:app --reload`"
            )
            st.stop()

        except requests.exceptions.Timeout:
            st.error(
                "Verification timed out. Try using a shorter "
                "video or check the backend."
            )
            st.stop()

        except requests.exceptions.RequestException as error:
            st.error(f"Request failed: {error}")
            st.stop()

    if response.status_code != 200:
        try:
            response_data = response.json()
            error = response_data.get(
                "detail",
                "Verification failed.",
            )

            if isinstance(error, dict):
                message = error.get(
                    "message",
                    "Verification failed.",
                )

                st.error(message)

                liveness_decision = error.get(
                    "liveness_decision"
                )

                liveness_score = error.get(
                    "liveness_score"
                )

                spoof_hint = error.get("spoof_hint")
                reason = error.get("reason")
                frames_used = error.get("frames_used")

                if liveness_decision:
                    st.write(
                        "**Liveness decision:**",
                        liveness_decision,
                    )

                if liveness_score is not None:
                    st.write(
                        "**Liveness score:**",
                        f"{float(liveness_score):.4f}",
                    )

                if spoof_hint:
                    st.write(
                        "**Possible reason:**",
                        spoof_hint.replace("_", " ").title(),
                    )

                if reason:
                    st.write(
                        "**Reason:**",
                        reason.replace("_", " ").title(),
                    )

                if frames_used is not None:
                    st.write(
                        "**Face frames detected:**",
                        frames_used,
                    )

                components = error.get("components")

                if components:
                    st.write("### Liveness Components")

                    component_columns = st.columns(
                        len(components)
                    )

                    for column, (
                        component_name,
                        component_value,
                    ) in zip(
                        component_columns,
                        components.items(),
                    ):
                        with column:
                            st.metric(
                                component_name,
                                f"{float(component_value):.4f}",
                            )

            else:
                st.error(str(error))

        except Exception:
            st.error(
                f"Verification failed with status "
                f"{response.status_code}."
            )

        st.stop()

    data = response.json()

    liveness_data = data.get("liveness", {})
    face_data = data.get("face_verification", {})

    liveness_verified = data.get(
        "liveness_verified",
        False,
    )

    matched = face_data.get(
        "matched",
        False,
    )

    similarity_score = float(
        face_data.get(
            "similarity_score",
            0.0,
        )
    )

    returned_threshold = float(
        face_data.get(
            "threshold",
            threshold,
        )
    )

    st.divider()
    st.header("Verification Result")

    if liveness_verified:
        st.success("✅ Liveness verification passed.")
    else:
        st.error("❌ Liveness verification failed.")
        st.stop()

    st.subheader("Liveness Analysis")

    l1, l2, l3, l4 = st.columns(4)

    with l1:
        st.metric(
            "Decision",
            liveness_data.get(
                "decision",
                "Unknown",
            ),
        )

    with l2:
        st.metric(
            "Liveness Score",
            f"{float(liveness_data.get('score', 0.0)):.4f}",
        )

    with l3:
        st.metric(
            "Frames Used",
            liveness_data.get(
                "frames_used",
                0,
            ),
        )

    with l4:
        bpm = float(
            liveness_data.get(
                "bpm",
                0.0,
            )
            or 0.0
        )

        st.metric(
            "Estimated BPM",
            f"{bpm:.2f}" if bpm > 0 else "Unavailable",
        )

    l5, l6, l7 = st.columns(3)

    with l5:
        st.metric(
            "Blink Count",
            liveness_data.get(
                "blink_count",
                0,
            ),
        )

    with l6:
        st.metric(
            "Mouth Open Count",
            liveness_data.get(
                "mouth_open_count",
                0,
            ),
        )

    with l7:
        mode = liveness_data.get(
            "mode",
            "Unknown",
        )

        st.metric(
            "Detection Mode",
            (
                mode.replace("_", " ").title()
                if mode
                else "Unknown"
            ),
        )

    components = liveness_data.get(
        "components",
        {},
    )

    if components:
        st.write("### Liveness Component Scores")

        component_columns = st.columns(
            len(components)
        )

        for column, (
            component_name,
            component_value,
        ) in zip(
            component_columns,
            components.items(),
        ):
            with column:
                st.metric(
                    component_name,
                    f"{float(component_value):.4f}",
                )

    st.divider()
    st.subheader("Face Matching Analysis")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.metric(
            "Similarity Score",
            f"{similarity_score:.4f}",
        )

    with f2:
        st.metric(
            "Threshold",
            f"{returned_threshold:.2f}",
        )

    with f3:
        st.metric(
            "Identity Status",
            "MATCHED ✅" if matched else "NOT MATCHED ❌",
        )

    st.progress(
        max(
            0.0,
            min(similarity_score, 1.0),
        )
    )

    st.divider()

    if matched:
        st.success(
            "🎉 Liveness passed and identity verified successfully!"
        )
        st.balloons()
    else:
        st.error(
            "❌ Liveness passed, but the face did not match "
            "the Government ID."
        )

    st.write("### Message")
    st.info(
        data.get(
            "message",
            "Verification completed.",
        )
    )

    if similarity_score >= returned_threshold:
        st.success(
            f"Similarity score ({similarity_score:.4f}) "
            f"is above the threshold "
            f"({returned_threshold:.2f})."
        )
    else:
        st.warning(
            f"Similarity score ({similarity_score:.4f}) "
            f"is below the threshold "
            f"({returned_threshold:.2f})."
        )

    spoof_hint = liveness_data.get("spoof_hint")

    if spoof_hint and spoof_hint != "unknown":
        st.warning(
            "Liveness information: "
            + spoof_hint.replace("_", " ").title()
        )
```
