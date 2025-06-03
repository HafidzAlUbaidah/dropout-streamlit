# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Latar Belakang:
Jaya Jaya Institut merupakan institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan menghasilkan banyak lulusan berkualitas. Namun, tingginya tingkat dropout (DO) menjadi isu krusial yang dapat mempengaruhi reputasi dan akreditasi institusi. Oleh karena itu, manajemen berinisiatif untuk mendeteksi potensi dropout sejak dini menggunakan pendekatan data-driven agar dapat memberikan intervensi tepat waktu.

### Permasalahan Bisnis

Tingginya jumlah mahasiswa yang tidak menyelesaikan pendidikan (dropout).

Tidak adanya sistem prediksi dini untuk mengidentifikasi siswa berisiko DO.

Kurangnya visualisasi dan alat monitoring performa akademik siswa yang intuitif.

### Cakupan Proyek

Melakukan eksplorasi dan analisis terhadap data performa akademik mahasiswa.

Mengembangkan dashboard interaktif sebagai alat monitoring performa mahasiswa.

Membangun prototipe sistem prediksi dropout berbasis machine learning.

Memberikan rekomendasi strategis berbasis data.

### Persiapan

Sumber data: Dataset "Student Performance" dari Jaya Jaya Institut, berisi informasi demografis, performa akademik semester 1 & 2, hingga status kelulusan mahasiswa.
Contoh atribut penting:

Curricular_units_1st_sem_grade dan Curricular_units_2nd_sem_grade: Indikator utama performa akademik.

Status: Target label (Graduate, Enrolled, Dropout).

Admission_grade, Previous_qualification_grade, Age_at_enrollment, dsb.

Setup environment:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Business Dashboard

Dashboard dibuat menggunakan Looker Studio yang menampilkan insight sebagai berikut:

Distribusi status mahasiswa (Graduate, Enrolled, Dropout).

Korelasi antar nilai akademik semester 1 dan semester 2.

Rata-rata nilai akademik berdasarkan status kelulusan.

Visualisasi performa tiap semester yang dikelompokkan berdasarkan target.

Link dashboard:
🔗 [https://lookerstudio.google.com/reporting/72f99308-6426-4dd0-80ec-d8b6e8e15141]

## Menjalankan Sistem Machine Learning

Sistem prediksi dikembangkan menggunakan algoritma Random Forest Classifier karena memberikan performa akurasi dan interpretasi yang baik. Model di-training menggunakan GridSearchCV dan disimpan dalam format .joblib.

Fitur penting:

Nilai akademik semester 1 & 2

Status pembayaran dan utang

Beberapa variabel sosiodemografis

Cara menjalankan prototype secara lokal:

streamlit run app.py

Link untuk menjalankan prototype di Streamlit Cloud:
🚀 [https://dropout-app-wkmapppktbbhvxglvxhgnrl.streamlit.app/]

## Conclusion

Melalui eksplorasi data, dashboard interaktif, dan model prediksi dropout, dapat disimpulkan bahwa:

Mahasiswa dengan performa buruk di semester awal berisiko tinggi untuk dropout.

Status keuangan (utang dan keterlambatan pembayaran) juga berkontribusi signifikan terhadap risiko DO.

Model machine learning yang dibangun mampu mengidentifikasi mahasiswa berisiko dengan akurasi cukup baik dan siap digunakan sebagai alat monitoring otomatis.

### Rekomendasi Action Items

Implementasi sistem notifikasi dini bagi mahasiswa yang terdeteksi berisiko tinggi oleh sistem ML.

Pendampingan akademik dan psikologis bagi mahasiswa dengan performa akademik rendah.

Evaluasi dan intervensi kebijakan pembayaran terhadap mahasiswa yang menunggak biaya pendidikan.

Peningkatan monitoring dashboard oleh staf akademik setiap semester.

Pembuatan program remedial akademik berbasis analisis historis dari dashboard performa mahasiswa.
