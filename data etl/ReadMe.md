## Dataset

Hasil dataset dapat dilihat pada [Dataset](https://drive.google.com/drive/folders/1McI8nlHV-b66qySo-BVqkQ69yds2_u57?usp=sharing)

Informasi Dataset Mountain
| Kolom            | Tipe Data | Deskripsi                                                                                |
| ---------------- | --------- | -----------------------------------------------------------------------------------------|
| `nama`           | `object`  | Nama gunung                                                                              |
| `provinsi`       | `object`  | Provinsi tempat gunung tersebut berada                                                   |
| `kabupaten`      | `object`  | Kabupaten/kota tempat gunung tersebut berada                                             |
| `kecamatan`      | `object`  | Kecamatan lokasi gunung                                                                  |
| `ketinggian`     | `int64`   | Ketinggian puncak gunung dari permukaan laut (dalam meter)                               |
| `jenis gunung`   | `object`  | Jenis gunung berdasarkan bentuk atau asal geologi (contoh: stratovolcano)                |
| `status`         | `object`  | Status aktivitas vulkanik (aktif atau tidak aktif)                                       |
| `akses`          | `object`  | Keterangan mengenai akses buka dan tutup gunung                                          |
| `jarak`          | `int64`   | Jarak pendakian dari titik awal sampai puncak (dalam meter)                              |
| `elevation gain` | `int64`   | Total kenaikan elevasi selama pendakian (dalam meter)                                    |
| `latitude`       | `float64` | Koordinat garis lintang gunung (untuk pemetaan geografis)                                |
| `longitude`      | `float64` | Koordinat garis bujur gunung (untuk pemetaan geografis)                                  |

<br>

Informasi Dataset Weather
| Kolom            | Tipe Data        | Deskripsi                                                            |
| ---------------- | ---------------- | ---------------------------------------------------------------------|
| `datetime`       | `datetime64[ns]` | Tanggal pengamatan cuaca dalam format `YYYY-MM-DD`                   |
| `datetimeEpoch`  | `int64`          | Tanggal dalam format epoch timestamp (detik sejak 1970-01-01)        |
| `tempmax`        | `float64`        | Suhu maksimum harian                                                 |
| `tempmin`        | `float64`        | Suhu minimum harian                                                  |
| `temp`           | `float64`        | Suhu rata-rata harian                                                |
| `feelslikemax`   | `float64`        | Suhu maksimum yang dirasakan                                         |
| `feelslikemin`   | `float64`        | Suhu minimum yang dirasakan                                          |
| `feelslike`      | `float64`        | Suhu rata-rata yang dirasakan                                        |
| `dew`            | `float64`        | Titik embun, suhu di mana kelembapan mulai terkondensasi             |
| `humidity`       | `float64`        | Kelembapan rata-rata                                                 |
| `precip`         | `float64`        | Curah hujan                                                          |
| `precipprob`     | `float64`        | Probabilitas terjadinya hujan                                        |
| `precipcover`    | `float64`        | Persentase wilayah yang mengalami hujan                              |
| `snow`           | `float64`        | Curah salju                                                          |
| `snowdepth`      | `float64`        | Kedalaman salju                                                      |
| `windgust`       | `float64`        | Kecepatan angin kencang maksimum                                     |
| `windspeed`      | `float64`        | Kecepatan angin rata-rata                                            |
| `winddir`        | `float64`        | Arah angin                                                           |
| `pressure`       | `float64`        | Tekanan udara                                                        |
| `cloudcover`     | `float64`        | Persentase tutupan awan                                              |
| `visibility`     | `float64`        | Jarak pandang                                                        |
| `solarradiation` | `float64`        | Intensitas radiasi matahari                                          |
| `solarenergy`    | `float64`        | Energi matahari per hari                                             |
| `uvindex`        | `float64`        | Indeks sinar ultraviolet                                             |
| `sunrise`        | `object`         | Waktu matahari terbit                                                |
| `sunriseEpoch`   | `int64`          | Waktu matahari terbit dalam epoch                                    |
| `sunset`         | `object`         | Waktu matahari terbenam                                              |
| `sunsetEpoch`    | `int64`          | Waktu matahari terbenam dalam epoch                                  |
| `moonphase`      | `float64`        | Fase bulan                                                           |
| `conditions`     | `object`         | Kondisi cuaca umum (contoh: `rain, partially cloudy`)                |
| `description`    | `object`         | Deskripsi rinci cuaca (misal: "partly cloudy throughout the day...") |
| `icon`           | `object`         | Gambar ikon representasi cuaca                                       |
| `stations`       | `object`         | Stasiun pengamatan cuaca                                             |
| `source`         | `object`         | Sumber data (contoh: `obs` untuk observasi langsung)                 |
| `kecamatan`      | `object`         | Nama kecamatan tempat data diambil                                   |
| `severerisk`     | `float64`        | Risiko cuaca ekstrem                                                 |
| `day`            | `int32`          | Tanggal dalam bulan (`1–31`)                                         |
| `month`          | `int32`          | Bulan (`1–12`)                                                       |
| `weekday`        | `int32`          | Hari dalam minggu (`0=Senin`, `6=Minggu`)                            |
| `year`           | `int32`          | Tahun (`YYYY`)                                                       |

<br>

Informasi Dataset Model 2
| Kolom              | Tipe Data   | Deskripsi                         |
| ------------------ | ------------| ----------------------------------|
| `ketinggian`       | `int64`     | Ketinggian jalur (mdpl)           |
| `jarak`            | `int64`     | Panjang jalur (meter)             |
| `elevation_gain`   | `int64`     | Kenaikan elevasi (mdpl)           |
| `temp`             | `float64`   | Suhu rata-rata harian (Celsius)   |
| `humidity`         | `float64`   | Kelembapan relatif (%)            |
| `precipprob`       | `float64`   | Probabilitas curah hujan (%)      |
| `windspeed`        | `float64`   | Kecepatan angin rata-rata (km/jam)|
| `difficulty_score` | `float64`   | Skor kesulitan jalur              |
| `estimated_time`   | `float64`   | Estimasi waktu tempuh             |

<br>

# Proses ETL
## Weather
### 1. Extract & Load (Raw)
Setiap anggota ML melakukan scraping weather dari website [Visual Crossing](https://www.visualcrossing.com/) untuk tahun yang berbeda dengan kecamatan sebagai lokasi yang diquery (list kecamatan didapat dari dataset mountain). Pembagian scraping dilakukan berdasarkan tahun:
- Vanessa: 2020, 2023
- Harry: 2024, 2025
- Faiza: 2021, 2022

Steps:
1. Open `scrape_weather_data.py`
2. Set:
   - input `YEAR` sesuai pembagian
   - input tanggal awal dan tanggal akhir di `START_DATE`, `END_DATE`
   - input API keys di `API_KEYS`
3. Run:
```bash
python scrape_weather_data.py
```

Hasil akan tersimpan dalam folder `YYYY/weather_per_region/` berdasarkan kecamatan masing dengan nama file `weather_{kecamatan}`

### 2. Merge & Clean
Steps:
1. Seluruh data hasil scraping di merge
2. Handle data null:
    - Datetime yang kosong diisi dengan date di kolom `wdatetime` (terdapat beberapa file yang menggunakan nama kolom `wdatetime`, namun fungsi kolom tersebut sama persis dengan `datetime`)
    - Kolom dengan data null yang banyak namun tidak memiliki informasi yang epnting di drop
    - Kolom dengan data null yang penting difill dengan foward fill and backward fill
3. Extract data day, month, weekday, dan year agar memperjelas tanggal, bulan, dan tahunnya
4. Lowercase seluruh data untuk menjaga konsistensi 
5. Save file ke dalam csv bernama `merged_weather`

## Mountain
### 1. Extract 
Data diambil secara manual dikarenakan tidak semua website memiliki data yang lengkap

### 2. Clean
Steps:
1. Menghapus kolom redundan seperti jarak memiliki 2 kolom, dalam km dan m (kolom km dihapus karena data lainnya menggunakan unit metrik m)
2. Menghapus kata seperti (m), (mdpl) pada nama kolom 
3. Lowercase seluruh data untuk menjaga konsistensi 
4. Parse koordinat menjadi 2 kolom berbeda, yaitu `latitude` dan `longitude`
5. Save file ke dalam csv bernama `gunung_indonesia`

## Model 2 Dataset
### 1. Extract & Load
Data diambil secara dari merge dataset weather dan mountain dengan `on='kecamatan', how='inner'` 

### 2. Clean
Steps:
1. Memilih features yang akan digunakan untuk membuat dataset baru  (`features = ['ketinggian', 'jarak', 'elevation gain', 'temp', 'humidity', 'precipprob', 'windspeed']`)
2. Cek data null dan duplikat
3. Drop data duplikat
4. Scale kolom yang diperlukan `'temp', 'humidity', 'precipprob', 'windspeed'` dengan `MinMaxScaler`

### 3. Create Dummy Targets
#### 1. Difficulty Score
```
def base_difficulty(row):
    elevation_gain_ft = row['elevation gain'] * 3.28084  # konversi meter ke feet
    distance_mi = row['jarak'] / 1609.34  # konversi meter ke mile
    raw_score = math.sqrt(elevation_gain_ft * 2 * distance_mi) #shenandoah's formula
    return raw_score ** 0.5

def full_difficulty_score(row):
    base = base_difficulty(row)
    weather_penalty = (
        0.25 * row['precipprob_scaled'] +
        0.25 * row['windspeed_scaled'] +
        0.25 * row['temp_scaled'] +
        0.25 * row['humidity_scaled']
    )
    SCALE_FACTOR = 0.25
    final_score = base * weather_penalty * SCALE_FACTOR
    return final_score

# Normalize difficulty to 1–10 range
def normalize_difficulty(scores):
    min_score = scores.min()
    max_score = scores.max()
    return 1 + 9 * ((scores - min_score) / (max_score - min_score))
```
Base Difficulty menggunakan rumus [Shenandoah's Hiking Difficulty Formula](https://www.nps.gov/shen/planyourvisit/how-to-determine-hiking-difficulty.htm), kemudian ditambah dengan weather penalty (model tidak hanya memprediksi berdasarkan elevation gain dan jarak, namun memperkirakan juga cuaca pada hari tersebut). Kemudian angka tersebut akan dinormalisasi menjadi 1-10.

#### 2. Estimated Time

```
def estimate_time(row):
    base_time = (row['jarak'] / 5000) + (row['elevation gain'] / 600) # naismith rule
    difficulty_factor = row['difficulty_score'] / 10 # scale 0–1
    return base_time * (1 + difficulty_factor)
```
Base Time menggunakan rumus [Naismith’s Rule](https://www.restless-viking.com/2018/11/29/naismiths-rule/) dimana setiap 5000 meter adalah 1 jam ditambah dengan 1 jam untuk 600 meter kenaikan elevasi, kemudian ditambah dengan hasil dari difficulty score untuk memprediksi estimasi waktu yang lebih dinamis dengan tingkat kesulitannya.

### 4. Save
Steps:
1. Drop kolom yang tidak digunakan lagi `'temp_scaled', 'precipprob_scaled', 'windspeed_scaled', 'humidity_scaled', 'base_difficulty_score', 'full_difficulty_score'`
2. Rename kolom `elevation gain` menjadi `elevation_gain` untuk konsistensi
3. Save ke file csv bernama `model2_dataset`
