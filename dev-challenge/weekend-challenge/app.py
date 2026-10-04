"""A reviewed workbench cleanup workflow powered by a local open-weight VLM."""

from hashlib import sha256
import json
from pathlib import Path

import streamlit as st

from workbench import MODEL, OBSERVATION_PROMPT, describe, make_checklist, normalize_image


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Workbench Reset", page_icon="🧰", layout="wide")
st.title("Shared Workbench Reset Assistant")
st.write("Compare with the reference, review what changed, and prepare a cleanup checklist.")
st.caption("Local open-weight AI · SmolVLM2 2.2B Q8 · Suggestions require your review")

source = st.radio("Choose photos", ["Demo sample", "My photos"], horizontal=True)
if source == "Demo sample":
    samples = sorted(path.name for path in (ROOT / "data").iterdir() if path.is_dir())
    sample = st.selectbox("Tabletop sample", samples)
    reference_data = (ROOT / "data" / sample / "frame000.png").read_bytes()
    current_data = (ROOT / "data" / sample / "frame004.png").read_bytes()
    source_label = f"TTU demo: {sample}"
    st.caption("Public TTU dataset samples; these are not photos of a coworker's workbench.")
else:
    left, right = st.columns(2)
    with left:
        reference_upload = st.file_uploader("Reference photo", type=["png", "jpg", "jpeg"], key="ref_upload")
    with right:
        current_upload = st.file_uploader("Current photo", type=["png", "jpg", "jpeg"], key="cur_upload")
    if reference_upload is None or current_upload is None:
        st.info("Choose both photos to continue.")
        st.stop()
    reference_data = reference_upload.getvalue()
    current_data = current_upload.getvalue()
    source_label = "User-provided photos"

try:
    reference_data = normalize_image(reference_data)
    current_data = normalize_image(current_data)
except (OSError, ValueError) as exc:
    st.error("Could not read a photo. Please choose valid PNG or JPEG images.")
    st.stop()

identity = sha256(reference_data + b"\x00" + current_data).hexdigest()
if st.session_state.get("image_identity") != identity:
    for key in ["observations", "reference_notes", "current_notes", "notes_confirmed",
                "checklist", "checklist_identity", "final_checklist", "final_confirmed"]:
        st.session_state.pop(key, None)
    st.session_state["image_identity"] = identity

left, right = st.columns(2)
with left:
    st.subheader("Reference: restore this layout")
    st.image(reference_data, use_container_width=True)
with right:
    st.subheader("Current: after use")
    st.image(current_data, use_container_width=True)

st.subheader("1. Observe")
if st.button("Analyze both photos", type="primary"):
    try:
        with st.spinner("Reading each photo independently on your laptop..."):
            reference_result = describe(reference_data)
            current_result = describe(current_data)
        st.session_state["observations"] = {"reference": reference_result, "current": current_result}
        st.session_state["reference_notes"] = reference_result["text"]
        st.session_state["current_notes"] = current_result["text"]
        st.session_state["notes_confirmed"] = False
        for key in ["checklist", "checklist_identity", "final_checklist", "final_confirmed"]:
            st.session_state.pop(key, None)
    except RuntimeError as exc:
        st.error(str(exc))

if "observations" not in st.session_state:
    st.stop()


def invalidate_notes():
    st.session_state["notes_confirmed"] = False
    for key in ["checklist", "checklist_identity", "final_checklist", "final_confirmed"]:
        st.session_state.pop(key, None)


st.subheader("2. Review observations")
st.write("Correct object names, positions, and orientations using the photos above. Keep uncertain details explicit.")
left, right = st.columns(2)
with left:
    reference_notes = st.text_area("Reference observations", key="reference_notes", height=180, on_change=invalidate_notes)
with right:
    current_notes = st.text_area("Current observations", key="current_notes", height=180, on_change=invalidate_notes)
for result in st.session_state["observations"].values():
    if result["finish_reason"] == "length":
        st.warning("An AI observation reached the output limit. Check for missing details.")
        break
notes_confirmed = st.checkbox("I checked both descriptions against the photos.", key="notes_confirmed")
notes_identity = sha256((reference_notes + "\x00" + current_notes).encode()).hexdigest()

st.subheader("3. Prepare the checklist")
if st.button("Draft cleanup checklist", disabled=not (notes_confirmed and reference_notes.strip() and current_notes.strip())):
    try:
        with st.spinner("Drafting from your reviewed observations..."):
            checklist = make_checklist(reference_notes, current_notes)
        st.session_state["checklist"] = checklist
        st.session_state["checklist_identity"] = notes_identity
        st.session_state["final_checklist"] = checklist["text"]
        st.session_state["final_confirmed"] = False
    except RuntimeError as exc:
        st.error(str(exc))

if "checklist" in st.session_state and st.session_state["checklist_identity"] == notes_identity:
    def invalidate_final():
        st.session_state["final_confirmed"] = False

    if st.session_state["checklist"]["finish_reason"] == "length":
        st.warning("The draft reached the output limit. Complete it before using it.")
    st.text_area("Edit the final checklist", key="final_checklist", height=200, on_change=invalidate_final)
    final_confirmed = st.checkbox("I checked the final instructions against the reference photo.", key="final_confirmed")
    if final_confirmed and notes_confirmed and st.session_state["final_checklist"].strip():
        st.success("Checklist ready for your review and use alongside the reference photo.")
        report = {
            "source": source_label,
            "model_alias": MODEL,
            "model_repository": "ggml-org/SmolVLM2-2.2B-Instruct-GGUF",
            "quantization": "Q8_0 model and projector",
            "backend": "llama.cpp local server",
            "image_sha256": {
                "reference": sha256(reference_data).hexdigest(),
                "current": sha256(current_data).hexdigest(),
            },
            "observation_prompt": OBSERVATION_PROMPT,
            "ai_observations": st.session_state["observations"],
            "reviewed_observations": {"reference": reference_notes, "current": current_notes},
            "ai_checklist": st.session_state["checklist"],
            "final_checklist": st.session_state["final_checklist"],
            "observations_confirmed_by_user": True,
            "checklist_confirmed_by_user": True,
            "note": "User-reviewed suggestions; no robot actions executed. Review free text for private information before sharing.",
        }
        st.download_button("Download reviewed result", json.dumps(report, indent=2),
                           file_name="workbench-reviewed-result.json", mime="application/json")
    st.caption("The downloaded result includes your text, but no photos, original filenames, or machine paths.")
