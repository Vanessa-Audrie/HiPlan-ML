import streamlit as st
import pandas as pd
import datetime
import requests
import gdown
import os

# Config API URLs
WEATHER_API_URL = "https://hiplan-ml-weather-pred.up.railway.app/forecast/seasonality"
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
 
# Ambil prediksi cuaca harian dari visual crossing
def get_weather_data(kecamatan):
    try:
        location_query = kecamatan.replace(" ", "%20")
        today = datetime.date.today().isoformat()
        api_key = "WCQPBLZHREA7LZ5RAX8MREF9N"
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location_query}?unitGroup=metric&key={api_key}&contentType=json&elements=datetime,temp,humidity,precipprob,windspeed,conditions&lang=id"

        res = requests.get(url)
        res.raise_for_status()
        return res.json().get("days", [])
    except Exception as e:
        st.error("❌ Gagal mengambil data cuaca.")
        st.text(str(e))
        return []

# Mapping condition ke bahasa Indonesia
condition_map = {
    "type_1": "Salju Kencang",
    "type_10": "Gerimis Beku Deras/Hujan Beku",
    "type_11": "Gerimis Beku Ringan/Hujan Beku",
    "type_12": "Kabut Beku",
    "type_13": "Hujan Beku Deras",
    "type_14": "Hujan Beku Ringan",
    "type_15": "Tornado/Awan Corong",
    "type_16": "Hujan Es",
    "type_17": "Es",
    "type_18": "Petir Tanpa Guntur",
    "type_19": "Kabut",
    "type_2": "Gerimis",
    "type_20": "Hujan Di Sekitar",
    "type_21": "Hujan",
    "type_22": "Hujan dan Salju Deras",
    "type_23": "Hujan dan Salju Ringan",
    "type_24": "Hujan Deras",
    "type_25": "Hujan Lebat",
    "type_26": "Hujan Ringan",
    "type_27": "Penurunan Coverage Awan",
    "type_28": "Kenaikan Coverage Awan",
    "type_29": "Coverage Awan Tidak Berubah",
    "type_3": "Gerimis Berat",
    "type_30": "Asap Atau Kabut",
    "type_31": "Salju",
    "type_32": "Salju dan Hujan Lebat",
    "type_33": "Hujan Salju",
    "type_34": "Salju Lebat",
    "type_35": "Salju Ringan",
    "type_36": "Angin Kencang",
    "type_37": "Hujan Badai",
    "type_38": "Badai Petir Tanpa Hujan",
    "type_39": "Debu Berlian",
    "type_4": "Gerimis Ringan",
    "type_40": "Hujan Es",
    "type_41": "Mendung",
    "type_42": "Sebagian Berawan",
    "type_43": "Cerah",
    "type_5": "Gerimis Berat/Hujan",
    "type_6": "Gerimis Ringan/Hujan",
    "type_7": "Badai Debu",
    "type_8": "Kabut",
    "type_9": "Gerimis Beku/Hujan Beku"
}

def translate_condition(condition_str):
    if not condition_str:
        return "Kondisi Tidak Diketahui"
    condition_ids = [c.strip() for c in condition_str.split(",")]
    condition_descriptions = []
    for cid in condition_ids:
        desc = condition_map.get(cid, "Kondisi Tidak Diketahui")
        condition_descriptions.append(desc)
    return ", ".join(condition_descriptions)

# Ambil Cuaca dari API Visual Crossing
st.markdown("<br><hr><br>", unsafe_allow_html=True)
st.header("Prediksi Cuaca 🌥️")
st.subheader("Cuaca Harian (Visual Crossing)")

if selected_gunung != "Pilih Gunung":
    cuaca_data = get_weather_data(gunung_data["Kecamatan"])
    st.subheader(f"Prediksi Cuaca 7 Hari di kecamatan {gunung_data['Kecamatan']}")

    if cuaca_data:
        st.session_state["cuaca_data"] = cuaca_data[:7]  # simpan 7 hari ke depan

        st.markdown("📌 *Klik tombol kecil di atas kartu cuaca untuk memilih hari yang diinginkan pada prediksi kesulitan dan estimasi waktu.*")

        cols = st.columns(7)
        for i, day in enumerate(cuaca_data[:7]):
            tanggal = datetime.datetime.strptime(day['datetime'], "%Y-%m-%d")
            hari = tanggal.strftime("%A").capitalize()
            tanggal_str = tanggal.strftime('%#d/%#m/%Y')

            kondisi = translate_condition(day.get("conditions", ""))

            konten = f"""
            <style>
            .card {{
                text-align:center; 
                padding:10px; 
                border-radius:10px; 
                cursor:pointer;
                height: 300px;   
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                background-color: #f1f2f6;
                color: #000000;
            }}

            @media (prefers-color-scheme: dark) {{
                .card {{
                background-color: #262730;
                color: #FFFFFF;
                }}
            }}
            </style>

            <div class="card">
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

# Model 1: Prediksi kecenderungan cuaca bulanan
st.markdown("<br>", unsafe_allow_html=True)
st.header("Kecenderungan Cuaca Bulanan")

if selected_gunung == "Pilih Gunung":
    st.info("📌 Silakan pilih gunung terlebih dahulu sebelum melihat prediksi kecenderungan cuaca.")
else:
    # Hanya muncul jika gunung sudah dipilih
    current_year = datetime.datetime.now().year
    months = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }

    month_options = ["Pilih bulan"] + list(months.keys())
    selected_month = st.selectbox("Bulan", options=month_options, format_func=lambda x: months[x] if isinstance(x, int) else x)

    year_options = ["Pilih tahun"] + list(range(current_year, current_year + 50))
    selected_year = st.selectbox("Tahun", options=year_options)

    if st.button("Lihat Prediksi Kecenderungan Cuaca Bulanan"):
        if selected_month == "Pilih bulan" or selected_year == "Pilih tahun":
            st.warning("Harap pilih bulan dan tahun terlebih dahulu.")
        else:
            params = {
                "kecamatan_name": gunung_data['Kecamatan'],
                "month": selected_month,
                "year": selected_year
            }
            try:
                response = requests.get(WEATHER_API_URL, params=params)
                response.raise_for_status()
                result = response.json()
                st.success("Hasil Prediksi")

                st.subheader("Rata-rata Parameter Cuaca:")
                metrics = result["analysis"]["reasoning_metrics"]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Suhu (°C)", f"{metrics['average_temp']:.2f} °C")
                col2.metric("Probabilitas Hujan (%)", f"{metrics['average_precipprob']:.2f} %")
                col3.metric("Kecepatan Angin", f"{metrics['average_windspeed']:.2f} km/h")
                col4.metric("Kelembaban (%)", f"{metrics['average_humidity']:.2f} %")

                st.subheader(f"Kecenderungan cuaca: **{result['analysis']['determined_seasonality']}**")

            except requests.exceptions.RequestException as e:
                st.error(f"Gagal menghubungi API cuaca: {e}")


# Model 2: Prediksi Kesulitan & Estimasi Waktu
st.markdown("<br><hr><br>", unsafe_allow_html=True)
st.header("Prediksi Kesulitan & Estimasi Waktu 🧭")

if "selected_day" not in st.session_state:
    st.info("📌 Pilih salah satu card prediksi cuaca harian dari atas untuk melanjutkan prediksi kesulitan.")
else:
    selected_day = st.session_state["selected_day"]

    st.markdown(f"*Tanggal Dipilih:* {selected_day['datetime']}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Suhu (°C)", f"{selected_day['temp']} °C")
    col2.metric("Probabilitas Hujan (%)", f"{selected_day['precipprob']}%")
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
                    f'<div style="{style}">🧭 Estimasi Waktu: {result.get("estimated_time", "-")}</div>',
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