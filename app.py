import streamlit as st
import pandas as pd
import joblib

# ------------------ Load Model dan Preprocessing ------------------
model = joblib.load('logreg_stress_model.pkl')
feature_names = joblib.load('feature_names.pkl')
scaler = joblib.load('scaler.pkl')

# ------------------ Tampilan Aplikasi ------------------
st.title("🎓 Prediksi Tingkat Stres Mahasiswa")
st.markdown("Masukkan data berikut untuk memprediksi tingkat stres berdasarkan aktivitas dan kondisi kamu:")

# ------------------ Input User ------------------
with st.form("stress_form"):
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    study_hours = st.slider("Jam belajar per minggu", 0, 50, 10)
    part_time = st.selectbox("Apakah kamu bekerja paruh waktu?", ["Ya", "Tidak"])
    extra_activities = st.selectbox("Ikut kegiatan ekstrakurikuler?", ["Ya", "Tidak"])
    social_support = st.selectbox("Punya dukungan sosial dari teman/keluarga?", ["Ya", "Tidak"])

    submit = st.form_submit_button("Submit")

# ------------------ Eksekusi Prediksi Saat Tombol Diklik ------------------
if submit:
    # Label Encoding Manual
    gender_enc = 1 if gender == "Perempuan" else 0
    part_time_enc = 1 if part_time == "Ya" else 0
    extra_activities_enc = 1 if extra_activities == "Ya" else 0
    social_support_enc = 1 if social_support == "Ya" else 0

    # Buat DataFrame Input
    input_dict = {
        'Gender': gender_enc,
        'Study Hours (hrs/week)': study_hours,
        'Part-time Job': part_time_enc,
        'Extracurricular Activities': extra_activities_enc,
        'Social Support': social_support_enc
    }

    input_data = pd.DataFrame([input_dict])
    input_data = input_data.reindex(columns=feature_names)

    # Validasi input
    if input_data.isnull().values.any():
        st.error("Data input tidak lengkap atau salah.")
    else:
        # Scaling
        scaled_input = scaler.transform(input_data)

        # Prediksi
        prediction = model.predict(scaled_input)[0]

        # Output hasil
        st.subheader("📊 Hasil Prediksi:")
        st.success(f"Tingkat stres kamu diprediksi adalah: **{prediction}**")
