import streamlit as st
import pandas as pd
import datetime
import requests
import gdown
import os

# Config API URLs
WEATHER_API_URL = "https://hiplan-ml-weather-pred.up.railway.app/forecast"
DIFFICULTY_API_URL = "https://hiplan-ml-diff-and-time-pred.up.railway.app/predict"
RECOMMENDER_API_URL = "https://hiplan-ml-recommender-system.up.railway.app/rekomendasi"

st.set_page_config(page_title="HiPlan", layout="wide")
st.title("HiPlan - Hike Safer, Plan Smart")

# Load Dataset dari Google Drive
@st.cache_data(show_spinner=True)
def load_gunung_data():
    file_url = "https://drive.google.com/uc?id=1z_sSo3VpzHZVhlQ-f7yLjPB7u9JOjHBx"
    output = "data_gunung.csv"
    if not os.path.exists(output):
        gdown.download(file_url, output, quiet=False)
    df = pd.read_csv(output)
    # Standarisasi nama kolom agar case-insensitive dan mudah dipanggil
    df.columns = [col.strip().capitalize() for col in df.columns]
    return df

df_gunung = load_gunung_data()

#  Pilih Gunung 
st.header("🔍 Pilih Gunung")

df_gunung["Nama_Display"] = df_gunung["Nama"].str.title()
gunung_list = ["Pilih Gunung"] + sorted(df_gunung["Nama_Display"].unique().tolist())
selected_gunung = st.selectbox("Gunung", gunung_list)

# Cek agar hanya menampilkan detail jika bukan opsi kosong
if selected_gunung != "Pilih Gunung":
    gunung_data = df_gunung[df_gunung["Nama_Display"] == selected_gunung].iloc[0]

    # Tampilkan info gunung
    st.markdown(f"""
    *Provinsi:* {gunung_data['Provinsi'].title()}  
    *Kecamatan:* {gunung_data['Kecamatan'].title()}  
    *Ketinggian:* {gunung_data['Ketinggian']} mdpl  
    *Akses:* {gunung_data['Akses'].capitalize()}
    """)

# Model 1: Prediksi Cuaca 
st.markdown("<br><hr><br>", unsafe_allow_html=True)
st.header("Prediksi Cuaca 🌤")

def get_weather_data(kecamatan):
    try:
        today = datetime.date.today().isoformat()
        payload = {
            "kecamatan_name": kecamatan,
            "start_date": today,
            "forecast_days": 7
        }

        response = requests.post(WEATHER_API_URL, json=payload)
        response.raise_for_status()

        return response.json()
    except Exception as e:
        st.error("❌ Gagal mengambil data cuaca dari API HiPlan.")
        st.text(str(e))
        return []

if selected_gunung != "Pilih Gunung":
    cuaca_data = get_weather_data(gunung_data["Kecamatan"])
    st.subheader(f"Prediksi Cuaca 7 Hari di kecamatan {gunung_data['Kecamatan']}")

    if cuaca_data:
        st.session_state["cuaca_data"] = cuaca_data[:7]  # simpan 7 hari ke depan

        st.markdown("📌 *Klik tombol kecil di atas kartu cuaca untuk memilih hari yang diinginkan.*")

        cols = st.columns(7)
        for i, day in enumerate(cuaca_data[:7]):
            tanggal = datetime.datetime.strptime(day['datetime'], "%Y-%m-%d")
            hari = tanggal.strftime("%A").capitalize()
            tanggal_str = tanggal.strftime('%#d/%#m/%Y')

            kondisi = "Cerah"
            if day['precipprob'] >= 80:
                kondisi = "Hujan"
            elif day['precipprob'] >= 50:
                kondisi = "Berawan"

            konten = f"""
                <div style="text-align:center; padding:10px; border-radius:10px; background-color:#262730; cursor:pointer;">
                    <h5 style="margin-bottom:0; margin-left:30px;">{hari}</h5>
                    <p style="margin:0;">{tanggal_str}</p>
                    <h3 style="margin-top:10px; margin-left:30px;">{day['temp']}°C</h3>
                    <p style="margin:0;">{kondisi}</p>
                    <p style="margin:0;">🌬 {day['windspeed']} km/h</p>
                    <p style="margin:0;">💧 {day['humidity']}%</p>
                    <p style="margin:0;">☔ {day['precipprob']}%</p>
                </div>
            """

            with cols[i]:
                if st.button(label="", key=f"card_{i}"):
                    st.session_state["selected_day"] = day
                st.markdown(konten, unsafe_allow_html=True)
            
else:
    st.info("📌 Pilih salah satu gunung dari atas untuk melanjutkan prediksi cuaca.")
    

# Model 2: Prediksi Kesulitan & Estimasi Waktu
st.markdown("<br><hr><br>", unsafe_allow_html=True)
st.header("Prediksi Kesulitan & Estimasi Waktu 🧭")

if "selected_day" not in st.session_state:
    st.info("📌 Pilih salah satu prediksi cuaca dari atas untuk melanjutkan prediksi kesulitan.")
else:
    selected_day = st.session_state["selected_day"]

    st.markdown(f"*Tanggal Dipilih:* {selected_day['datetime']}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperatur", f"{selected_day['temp']} °C")
    col2.metric("Kemungkinan Hujan (%)", f"{selected_day['precipprob']}%")
    col3.metric("Kecepatan Angin", f"{selected_day['windspeed']} km/h")
    col4.metric("Kelembapan (%)", f"{selected_day['humidity']}%")

    elevation_gain = float(gunung_data.get("Elevation gain", 0))
    jarak = float(gunung_data.get("Jarak", 0))
    ketinggian = float(gunung_data["Ketinggian"])

    st.markdown(f"""
    📌 *Data Gunung:* 
    - Ketinggian: {int(ketinggian)} mdpl  
    - Jarak: {int(jarak)} m  
    - Elevation Gain: {int(elevation_gain)} mdpl  
    """)

    if "selected_day" in st.session_state:
        payload = {
            "ketinggian": ketinggian,
            "jarak": jarak,
            "elevation_gain": elevation_gain,
            "temp": selected_day["temp"],
            "precipprob": selected_day["precipprob"],
            "windspeed": selected_day["windspeed"],
            "humidity": selected_day["humidity"]
        }

        try:            
            response = requests.post(DIFFICULTY_API_URL, json=payload)
            response.raise_for_status()

            if response.status_code != 200:
                st.error(f"❌ Gagal memprediksi. Status Code: {response.status_code}")
                st.text(f"Detail: {response.text}")
            else:
                result = response.json()
                style = "margin-bottom: 4px; font-weight: bold; color: #FFFFFF; font-size: 18px;"
                st.success("Hasil Prediksi")
                st.markdown(
                    f'<div style="{style}">🧗 Skor Kesulitan: {result.get("difficulty_score", "-")}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div style="{style}">⏱ Estimasi Waktu: {result.get("estimated_time", "-")}</div>',
                    unsafe_allow_html=True
                )
            
        except requests.exceptions.RequestException as e:
            st.error("Gagal memprediksi.")
            st.text(str(e))

# Model 3: Rekomendasi Gunung 
st.markdown("<br><hr><br>", unsafe_allow_html=True)
st.header("Rekomendasi Gunung 🗻")

with st.form(key="recommendation_form"):
    lokasi = st.text_input("📍*Lokasi Pendakian (Kecamatan/Provinsi)*")
    ketinggian = st.number_input("🚩*Ketinggian Target (meter)*", min_value=0)
    submit_recommend = st.form_submit_button("Cari Rekomendasi")

if submit_recommend:
    rekom_input = {
        "lokasi": lokasi,
        "ketinggian": ketinggian
    }

    try:
        response = requests.post(RECOMMENDER_API_URL, json=rekom_input)
        response.raise_for_status()
        raw_json = response.json()
        rekom_result = raw_json.get("rekomendasi", [])

        if not rekom_result:
            st.warning("Tidak ditemukan rekomendasi berdasarkan input Anda.")
        else:
            st.success("Berikut rekomendasi gunung yang sesuai:")
            for gunung in rekom_result:
                st.markdown(f"""
                <div style="padding: 10px;">
                    <p style="margin-bottom: 4px; font-weight: bold; color: #8FC098; font-size: 18px;">
                        {gunung.get('Nama', '-')}
                    </p>
                    <p style="margin: 0;">Provinsi: {gunung.get('Provinsi', '-')}</p>
                    <p style="margin: 0;">Ketinggian: {gunung.get('Ketinggian (dpl)', '-')} mdpl</p>
                    <p style="margin: 0;">Akses: {gunung.get('Akses', '-')}</p>
                    <hr>
                </div>
    """, unsafe_allow_html=True)
                
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Gagal memuat rekomendasi: {e}")
        with st.expander("Detail kesalahan (debug)"):
            st.json(rekom_input, expanded=False)
            st.text(f"Status code: {response.status_code if 'response' in locals() else 'N/A'}")
            st.text("Response content:")
            st.text(response.text if 'response' in locals() else str(e))