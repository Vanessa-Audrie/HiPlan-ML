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
Steps:
1. Data diambil secara manual dikarenakan tidak semua website memiliki data yang lengkap
2. Cleaning:
    - Menghapus kolom redundan seperti jarak memiliki 2 kolom, dalam km dan m (kolom km dihapus karena data lainnya menggunakan unit metrik m)
    - Menghapus kata seperti (m), (mdpl) pada nama kolom 
    - Lowercase seluruh data untuk menjaga konsistensi 
    - Parse koordinat menjadi 2 kolom berbeda, yaitu `latitude` dan `longitude`
3. Save file ke dalam csv bernama `gunung_indonesia`

