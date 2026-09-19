import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from deepface import DeepFace

st.set_page_config(page_title="AI Face Recognition Attendance", page_icon="🎓", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
STUDENT_FILE = BASE_DIR / "students.csv"
ATTENDANCE_FILE = BASE_DIR / "attendance.csv"
DATABASE_FOLDER = BASE_DIR / "database"
ATTENDANCE_COLUMNS = ["name", "roll_no", "date", "time", "status"]

def load_students():
    if not STUDENT_FILE.exists():
        st.error("students.csv was not found.")
        st.stop()
    try:
        df = pd.read_csv(STUDENT_FILE)
    except Exception as exc:
        st.error(f"Unable to read students.csv: {exc}")
        st.stop()
    missing = {"name", "roll_no"} - set(df.columns)
    if missing:
        st.error(f"students.csv is missing: {', '.join(sorted(missing))}")
        st.stop()
    return df

def load_attendance():
    if not ATTENDANCE_FILE.exists() or ATTENDANCE_FILE.stat().st_size == 0:
        df = pd.DataFrame(columns=ATTENDANCE_COLUMNS)
        df.to_csv(ATTENDANCE_FILE, index=False)
        return df
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        df = pd.DataFrame(columns=ATTENDANCE_COLUMNS)
    for col in ATTENDANCE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[ATTENDANCE_COLUMNS]

def mark_attendance(student_name, students):
    matched = students[students["name"].astype(str).str.strip().str.casefold() == student_name.strip().casefold()]
    if matched.empty:
        return None, "Face recognized, but the student is not registered."

    student = matched.iloc[0]
    name, roll_no = str(student["name"]), str(student["roll_no"])
    now = datetime.now()
    date, time = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
    attendance = load_attendance()

    existing = attendance[(attendance["roll_no"].astype(str) == roll_no) & (attendance["date"].astype(str) == date)]
    if not existing.empty:
        return {"name": name, "roll_no": roll_no, "date": date,
                "time": existing.iloc[0]["time"], "status": "Already Present"}, None

    row = {"name": name, "roll_no": roll_no, "date": date, "time": time, "status": "Present"}
    attendance = pd.concat([attendance, pd.DataFrame([row])], ignore_index=True)
    attendance.to_csv(ATTENDANCE_FILE, index=False)
    return row, None

st.title("🎓 AI Face Recognition Attendance System")
st.caption("Automatic student identification and attendance tracking using DeepFace.")

students = load_students()

with st.sidebar:
    st.header("📌 Project Information")
    st.write(f"**Registered Students:** {len(students)}")
    st.write("**Model:** ArcFace")
    st.write("**Detector:** OpenCV")
    st.write("**Framework:** Streamlit")
    if st.button("🔄 Refresh Attendance"):
        st.rerun()

st.subheader("📸 Recognize Student")
uploaded_file = st.file_uploader(
    "Upload a student's photo", type=["jpg", "jpeg", "png"],
    help="Upload a clear photo containing one registered student's face."
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Photo", width=300)
    if st.button("🔍 Recognize & Mark Attendance", type="primary"):
        if not DATABASE_FOLDER.exists():
            st.error("Database folder was not found.")
            st.stop()

        temp_path = None
        try:
            suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            with st.spinner("🔍 Recognizing face..."):
                results = DeepFace.find(
                    img_path=temp_path,
                    db_path=str(DATABASE_FOLDER),
                    model_name="ArcFace",
                    detector_backend="opencv",
                    enforce_detection=True,
                    silent=True,
                )

            if not results or results[0].empty:
                st.error("❌ No matching registered student was found.")
            else:
                identity = results[0].iloc[0]["identity"]
                student_name = Path(str(identity)).parent.name
                record, error = mark_attendance(student_name, students)

                if error:
                    st.warning(f"⚠️ {error}")
                elif record:
                    if record["status"] == "Already Present":
                        st.warning(f"⚠️ {record['name']}'s attendance is already marked today.")
                    else:
                        st.success("✅ Attendance marked successfully!")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Student", record["name"])
                    c2.metric("Roll No.", record["roll_no"])
                    c3.metric("Date", record["date"])
                    c4.metric("Time", record["time"])
        except Exception as exc:
            st.error("❌ Face recognition failed.")
            st.info("Try a clear front-facing photo of a registered student.")
            with st.expander("Technical error"):
                st.code(str(exc))
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

st.divider()
st.subheader("📊 Attendance Records")
attendance = load_attendance()
if attendance.empty:
    st.info("No attendance records yet.")
else:
    st.dataframe(attendance, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download Attendance CSV",
                       data=attendance.to_csv(index=False).encode("utf-8"),
                       file_name="attendance.csv", mime="text/csv")
