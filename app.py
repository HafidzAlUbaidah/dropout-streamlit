import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 
from datetime import datetime 

# --- Konfigurasi Aplikasi ---
st.set_page_config(layout="wide", page_title="Analisis Prediksi Dropout Mahasiswa")
APP_VERSION = "1.8" # Anda bisa perbarui versi di sini

# --- Inisialisasi Session State untuk Timestamp ---
if 'model_load_time' not in st.session_state:
    st.session_state.model_load_time = None
if 'data_viz_load_time' not in st.session_state:
    st.session_state.data_viz_load_time = None
if 'model_status' not in st.session_state:
    st.session_state.model_status = "Belum Dimuat"


# --- Konfigurasi Path ---
DATA_PATH = "data/data.csv"
MODEL_PATH_JOBLIB = 'model/tuned_lightgbm_model.joblib'
MODEL_PATH_PKL = 'model/tuned_lightgbm_model.pkl'

# --- Muat Model Machine Learning Anda ---
@st.cache_resource 
def load_model_with_timestamp():
    model_obj = None
    try:
        model_obj = joblib.load(MODEL_PATH_JOBLIB)
        st.session_state.model_load_time = datetime.now()
        st.session_state.model_status = "Berhasil Dimuat"
        return model_obj
    except FileNotFoundError:
        try:
            model_obj = joblib.load(MODEL_PATH_PKL)
            st.session_state.model_load_time = datetime.now()
            st.session_state.model_status = "Berhasil Dimuat"
            return model_obj
        except FileNotFoundError:
            st.session_state.model_status = "File Model Tidak Ditemukan"
            st.session_state.model_load_time = None
            return None
        except Exception as e:
            st.session_state.model_status = f"Error Pemuatan Model (.pkl): {e}"
            st.session_state.model_load_time = None
            return None
    except Exception as e:
        st.session_state.model_status = f"Error Pemuatan Model (.joblib): {e}"
        st.session_state.model_load_time = None
        return None

model = load_model_with_timestamp() # Panggil fungsi yang sudah dimodifikasi

if st.session_state.model_status == "Berhasil Dimuat":
    st.sidebar.success(f"Model: {st.session_state.model_status}")
else:
    st.sidebar.error(f"Model: {st.session_state.model_status}")


# --- Fungsi untuk memuat dan membersihkan data (untuk visualisasi) ---
@st.cache_data 
def load_data_for_visualization_with_timestamp(data_path=DATA_PATH):
    try:
        df = pd.read_csv(data_path, sep=';')
        st.session_state.data_viz_load_time = datetime.now() # Catat waktu pemuatan data
        if 'Status' in df.columns:
            df['Status'] = df['Status'].str.strip()
        else:
            st.warning("Kolom 'Status' tidak ditemukan dalam dataset. Beberapa fitur visualisasi mungkin tidak berfungsi.")
        
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'[\(\)]', '', regex=True)
        
        potential_numeric_cols = [
            'Previous_qualification_grade', 'Admission_grade',
            'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_grade',
            'Age_at_enrollment', 'Unemployment_rate', 'Inflation_rate', 'GDP',
            'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
            'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
            'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
            'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
            'Application_order'
        ]
        for col in potential_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        st.session_state.data_viz_load_time = "File Data Tidak Ditemukan"
        st.error(f"File dataset tidak ditemukan di {data_path} untuk visualisasi.")
        return pd.DataFrame()
    except Exception as e:
        st.session_state.data_viz_load_time = f"Error Pemuatan Data: {e}"
        st.error(f"Error saat memuat data untuk visualisasi: {e}")
        return pd.DataFrame()

# --- Navigasi Aplikasi ---
st.sidebar.title("Menu Navigasi")
page_options = ["📊 Dashboard Analisis Data", "🤖 Prediksi Status Mahasiswa (ML)"]
page = st.sidebar.radio("Pilih Halaman:", page_options, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("**Jaya Jaya Institut**")
st.sidebar.markdown("🎓 *Solusi Deteksi Dini Mahasiswa Dropout*")

# --- Halaman Visualisasi Data ---
if page == "📊 Dashboard Analisis Data":
    st.title("📊 Dashboard Analisis Data Mahasiswa")
    st.markdown("Eksplorasi data mahasiswa untuk mendapatkan wawasan terkait faktor-faktor yang mempengaruhi status kelulusan.")

    # Panggil fungsi pemuatan data yang mencatat timestamp
    df_viz = load_data_for_visualization_with_timestamp() 

    if not df_viz.empty and 'Status' in df_viz.columns:
        
        st.sidebar.header("Filter Data (Visualisasi)")
        categorical_cols_for_filter = [col for col in df_viz.columns if df_viz[col].nunique() < 20 and df_viz[col].dtype == 'object' and col != 'Status']
        
        filters = {}
        for col in categorical_cols_for_filter:
            unique_options = df_viz[col].dropna().unique()
            options = ['Semua'] + sorted(list(unique_options))
            filters[col] = st.sidebar.selectbox(f"Filter berdasarkan {col.replace('_', ' ').title()}:", options, index=0, key=f"filter_{col}")

        df_filtered = df_viz.copy()
        for col, val in filters.items():
            if val != 'Semua':
                df_filtered = df_filtered[df_filtered[col] == val]

        if df_filtered.empty:
            st.warning("Tidak ada data yang cocok dengan filter yang dipilih.")
        else:
            st.header("📈 Key Performance Indicators (KPIs)")
            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            
            total_students = len(df_filtered)
            dropout_count = df_filtered[df_filtered['Status'] == 'Dropout'].shape[0]
            
            if total_students > 0:
                dropout_rate = (dropout_count / total_students) * 100
                col_kpi1.metric("Tingkat Dropout Keseluruhan", f"{dropout_rate:.2f}%", f"{dropout_count} dari {total_students} mahasiswa")

                avg_admission_grade_dropout = df_filtered[df_filtered['Status'] == 'Dropout']['Admission_grade'].mean()
                avg_admission_grade_graduate = df_filtered[df_filtered['Status'] == 'Graduate']['Admission_grade'].mean()
                
                if pd.notna(avg_admission_grade_graduate):
                    col_kpi2.metric("Rata-rata Nilai Penerimaan (Lulus)", f"{avg_admission_grade_graduate:.2f}")
                if pd.notna(avg_admission_grade_dropout):
                     col_kpi2.metric("Rata-rata Nilai Penerimaan (Dropout)", f"{avg_admission_grade_dropout:.2f}", delta=f"{avg_admission_grade_dropout - avg_admission_grade_graduate:.2f} vs Lulus" if pd.notna(avg_admission_grade_graduate) else None, delta_color="inverse")

                avg_approved_1st_sem_graduate = df_filtered[df_filtered['Status'] == 'Graduate']['Curricular_units_1st_sem_approved'].mean()
                avg_approved_1st_sem_dropout = df_filtered[df_filtered['Status'] == 'Dropout']['Curricular_units_1st_sem_approved'].mean()
                if pd.notna(avg_approved_1st_sem_graduate):
                    col_kpi3.metric("Rata-rata SKS Lulus Sem 1 (Lulus)", f"{avg_approved_1st_sem_graduate:.2f}")
                if pd.notna(avg_approved_1st_sem_dropout):
                    col_kpi3.metric("Rata-rata SKS Lulus Sem 1 (Dropout)", f"{avg_approved_1st_sem_dropout:.2f}", delta_color="inverse")
            else:
                st.info("Tidak ada data untuk menampilkan KPI setelah filter diterapkan.")

            st.markdown("---")

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Distribusi Status", 
                "🎻 Distribusi Fitur Numerik", 
                "🧮 Analisis Fitur Kategorikal", 
                "🔗 Korelasi Fitur",
                "📄 Data Mentah Terfilter"
            ])

            with tab1:
                st.subheader("Distribusi Status Mahasiswa")
                if not df_filtered['Status'].empty:
                    status_counts = df_filtered['Status'].value_counts()
                    fig_status_pie = px.pie(status_counts, values=status_counts.values, names=status_counts.index,
                                            title='Proporsi Status Mahasiswa', hole=.3,
                                            color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_status_pie.update_layout(legend_title_text='Status')
                    st.plotly_chart(fig_status_pie, use_container_width=True)
                else:
                    st.info("Tidak ada data status untuk ditampilkan.")

            with tab2:
                st.subheader("Distribusi Fitur Numerik Utama berdasarkan Status")
                numeric_cols_for_violin = ['Admission_grade', 'Previous_qualification_grade', 'Age_at_enrollment', 'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_approved', 'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_approved']
                valid_numeric_cols_violin = [col for col in numeric_cols_for_violin if col in df_filtered.columns and df_filtered[col].nunique() > 1]

                if valid_numeric_cols_violin:
                    selected_violin_col = st.selectbox("Pilih Fitur Numerik untuk Violin Plot:", valid_numeric_cols_violin, key="violin_select")
                    if selected_violin_col:
                        fig_violin = px.violin(df_filtered, y=selected_violin_col, x='Status', color='Status',
                                               box=True, points="all", hover_data=df_filtered.columns,
                                               title=f"Distribusi {selected_violin_col.replace('_',' ').title()} berdasarkan Status",
                                               color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig_violin.update_layout(legend_title_text='Status')
                        st.plotly_chart(fig_violin, use_container_width=True)
                else:
                    st.info("Tidak ada fitur numerik yang valid untuk ditampilkan dalam violin plot.")

            with tab3:
                st.subheader("Analisis Fitur Kategorikal terhadap Status")
                candidate_cat_cols = [col for col in df_filtered.columns if df_filtered[col].dtype == 'object' and df_filtered[col].nunique() < 15 and col != 'Status']
                if candidate_cat_cols:
                    selected_cat_col_analysis = st.selectbox("Pilih Fitur Kategorikal untuk Analisis:", candidate_cat_cols, key="cat_analysis_select")
                    if selected_cat_col_analysis:
                        grouped_bar_data = df_filtered.groupby([selected_cat_col_analysis, 'Status']).size().reset_index(name='Jumlah')
                        fig_cat_bar = px.bar(grouped_bar_data, x=selected_cat_col_analysis, y='Jumlah', color='Status',
                                             barmode='group', title=f"Jumlah Mahasiswa berdasarkan {selected_cat_col_analysis.replace('_',' ').title()} dan Status",
                                             color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig_cat_bar.update_layout(legend_title_text='Status')
                        st.plotly_chart(fig_cat_bar, use_container_width=True)

                        df_percentage = df_filtered.groupby(selected_cat_col_analysis)['Status'].value_counts(normalize=True).mul(100).rename('Persentase').reset_index()
                        fig_cat_stacked_bar = px.bar(df_percentage, x=selected_cat_col_analysis, y='Persentase', color='Status',
                                             title=f"Persentase Status Mahasiswa berdasarkan {selected_cat_col_analysis.replace('_',' ').title()}",
                                             color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig_cat_stacked_bar.update_layout(legend_title_text='Status', yaxis_ticksuffix="%")
                        st.plotly_chart(fig_cat_stacked_bar, use_container_width=True)

                else:
                    st.info("Tidak ada fitur kategorikal yang cocok untuk analisis mendalam saat ini.")

            with tab4:
                st.subheader("Heatmap Korelasi Antar Fitur Numerik")
                numeric_df = df_filtered.select_dtypes(include=np.number) 
                if not numeric_df.empty and numeric_df.shape[1] > 1:
                    corr_matrix = numeric_df.corr()
                    fig_heatmap, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=.5, annot_kws={"size":8})
                    ax.set_title("Matriks Korelasi Fitur Numerik", fontsize=16)
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                    plt.yticks(rotation=0, fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig_heatmap)
                else:
                    st.info("Tidak cukup fitur numerik untuk membuat heatmap korelasi.")

            with tab5:
                st.subheader("Tabel Data Mahasiswa (Terfilter)")
                st.markdown("Anda dapat mengurutkan tabel dengan mengklik header kolom.")
                st.dataframe(df_filtered, use_container_width=True, height=500)

    elif df_viz.empty:
        st.error("Gagal memuat data. Silakan periksa path dan file dataset Anda.")
    else: 
        st.error("Kolom 'Status' tidak ditemukan dalam dataset. Fungsi visualisasi dashboard terbatas.")
        st.subheader("Pratinjau Data (Tanpa Kolom Status)")
        st.dataframe(df_viz.head())


# --- Halaman Prediksi ML ---
elif page == "🤖 Prediksi Status Mahasiswa (ML)":
    st.title("🤖 Prediksi Status Kelulusan Mahasiswa")

    if model is None:
        st.error("Model machine learning tidak berhasil dimuat. Fitur prediksi tidak tersedia.")
    else:
        st.markdown("""
        Masukkan data mahasiswa di bawah ini untuk mendapatkan prediksi status.
        Pastikan fitur dan tipe data yang Anda masukkan **SESUAI PERSIS** dengan yang diharapkan oleh model Anda.
        Nama kolom di DataFrame yang dikirim ke model harus sama persis dengan yang digunakan saat training.
        """)

        with st.form("student_input_form_final_v5"): 
            st.header("Form Input Data Mahasiswa")
            
            marital_status_options = {1: "Single", 2: "Married", 3: "Widower", 4: "Divorced", 5: "Facto Union", 6: "Legally Separated"}
            daytime_attendance_options = {1: "Daytime", 0: "Evening"}
            yes_no_options = {1: "Yes", 0: "No"}
            gender_options = {1: "Male", 0: "Female"}
            
            col_form_1, col_form_2, col_form_3 = st.columns(3)

            with col_form_1:
                st.subheader("Informasi Pribadi & Aplikasi")
                ms_key = st.selectbox("Status Pernikahan", list(marital_status_options.keys()), format_func=lambda x: f"{x} - {marital_status_options[x]}", key="ms_final_v5")
                app_mode = st.number_input("Kode Mode Aplikasi", min_value=1, value=17, help="Lihat kamus data untuk kode Application_mode", key="app_mode_final_v5")
                ap_order = st.number_input("Urutan Pilihan Aplikasi", min_value=0, max_value=9, value=1, help="0: Pilihan pertama", key="app_order_final_v5")
                age_enroll = st.number_input("Usia Saat Pendaftaran", min_value=16, max_value=70, value=20, key="age_final_v5")
                gender_key = st.selectbox("Jenis Kelamin", list(gender_options.keys()), format_func=lambda x: gender_options[x], key="gender_final_v5")
                nationality = st.number_input("Kode Kebangsaan", min_value=1, value=1, help="Lihat kamus data untuk Nacionality", key="nationality_final_v5")
                international_key = st.selectbox("Mahasiswa Internasional?", list(yes_no_options.keys()), format_func=lambda x: yes_no_options[x], key="international_final_v5")

            with col_form_2:
                st.subheader("Latar Belakang Pendidikan & Ekonomi")
                pq_key = st.number_input("Kode Kualifikasi Sebelumnya", min_value=1, value=1, help="Lihat kamus data untuk Previous_qualification", key="pq_final_v5")
                pq_grade = st.number_input("Nilai Kualifikasi Sebelumnya", min_value=0.0, max_value=200.0, value=120.0, step=0.1, key="pq_grade_final_v5")
                adm_grade = st.number_input("Nilai Penerimaan", min_value=0.0, max_value=200.0, value=125.0, step=0.1, key="adm_grade_final_v5")
                debtor_key = st.selectbox("Status Debitur?", list(yes_no_options.keys()), format_func=lambda x: yes_no_options[x], key="debtor_final_v5")
                tuition_fees_key = st.selectbox("Biaya Kuliah Lunas?", list(yes_no_options.keys()), format_func=lambda x: yes_no_options[x], key="tuition_final_v5")
                scholarship_key = st.selectbox("Pemegang Beasiswa?", list(yes_no_options.keys()), format_func=lambda x: yes_no_options[x], key="scholarship_final_v5")
                displaced_key = st.selectbox("Mahasiswa Pindahan (Displaced)?", list(yes_no_options.keys()), format_func=lambda x: yes_no_options[x], key="displaced_final_v5")
            
            with col_form_3:
                st.subheader("Detail Akademik Semester 1")
                course = st.number_input("Kode Program Studi", min_value=33, value=9773, help="Lihat kamus data untuk Course", key="course_final_v5")
                dt_attendance_key = st.selectbox("Waktu Kuliah (Siang/Malam)", list(daytime_attendance_options.keys()), format_func=lambda x: daytime_attendance_options[x], key="dt_attend_final_v5")
                edu_special_key = st.selectbox("Kebutuhan Pddkn. Khusus?", list(yes_no_options.keys()), format_func=lambda x: yes_no_options[x], key="edu_special_final_v5")
                cu1_credited = st.number_input("SKS Diakui Sem 1", min_value=0, value=0, key="cu1_cred_final_v5")
                cu1_enrolled = st.number_input("SKS Diambil Sem 1", min_value=0, max_value=50, value=6, key="cu1_enr_final_v5")
                cu1_evals = st.number_input("Jumlah Evaluasi Sem 1", min_value=0,value=8, key="cu1_eval_final_v5")
                cu1_approved = st.number_input("SKS Lulus Sem 1", min_value=0, max_value=50,value=6, key="cu1_appr_final_v5")
                cu1_grade = st.number_input("Rata-rata Nilai Sem 1", min_value=0.0, max_value=20.0, value=13.5, step=0.01, key="cu1_grade_final_v5", help="Skala 0-20")

            with st.expander("Latar Belakang Keluarga "):
                col_parent_1, col_parent_2 = st.columns(2)
                with col_parent_1:
                    mother_qual = st.number_input("Kode Kualifikasi Ibu", min_value=1, value=37, help="Lihat kamus data", key="mom_qual_final_v5")
                    father_qual = st.number_input("Kode Kualifikasi Ayah", min_value=1, value=37, help="Lihat kamus data", key="dad_qual_final_v5")
                with col_parent_2:
                    mother_occup = st.number_input("Kode Pekerjaan Ibu", min_value=0, value=9, help="Lihat kamus data", key="mom_occup_final_v5")
                    father_occup = st.number_input("Kode Pekerjaan Ayah", min_value=0, value=9, help="Lihat kamus data", key="dad_occup_final_v5")

            with st.expander("Detail Akademik Semester 2"):
                col_sem2_1, col_sem2_2, col_sem2_3, col_sem2_4 = st.columns(4)
                with col_sem2_1:
                    cu2_credited = st.number_input("SKS Diakui Sem 2", min_value=0, value=0, key="cu2_cred_final_v5")
                with col_sem2_2:
                    cu2_enrolled = st.number_input("SKS Diambil Sem 2", min_value=0, max_value=50, value=6, key="cu2_enr_final_v5")
                with col_sem2_3:
                    cu2_evals = st.number_input("Jumlah Evaluasi Sem 2", min_value=0, value=6, key="cu2_eval_final_v5")
                with col_sem2_4:
                    cu2_approved = st.number_input("SKS Lulus Sem 2", min_value=0, max_value=50, value=5, key="cu2_appr_final_v5")
                cu2_grade = st.number_input("Rata-rata Nilai Sem 2", min_value=0.0, max_value=20.0, value=12.0, step=0.01, key="cu2_grade_final_v5", help="Skala 0-20")
            
            with st.expander("Fitur Ekonomi"):
                unemployment_rate_val = st.number_input("Tingkat Pengangguran (Unemployment rate)", value = 10.0, step=0.1, key="unemp_rate_final_v5")
                inflation_rate_val = st.number_input("Tingkat Inflasi (Inflation rate)", value = 1.0, step=0.1, key="inf_rate_final_v5")
                gdp_val = st.number_input("GDP", value=0.0, key="gdp_final_v5", step=0.01)


            submit_button = st.form_submit_button(label="🚀 Prediksi Status & Analisis")

        if submit_button:
            pass_ratio_sem1_val = (cu1_approved / cu1_enrolled) if cu1_enrolled > 0 else 0.0
            pass_ratio_sem2_val = (cu2_approved / cu2_enrolled) if cu2_enrolled > 0 else 0.0
            total_enrolled_val = float(cu1_enrolled + cu2_enrolled) 
            
            grades_sum = 0.0
            semesters_with_grades = 0
            if cu1_enrolled > 0 and cu1_grade is not None:
                grades_sum += float(cu1_grade)
                semesters_with_grades += 1
            if cu2_enrolled > 0 and cu2_grade is not None:
                grades_sum += float(cu2_grade)
                semesters_with_grades += 1
            average_grade_val = (grades_sum / semesters_with_grades) if semesters_with_grades > 0 else 0.0

            input_data_dict_temp = {
                'Marital_status': ms_key, 'Application_mode': app_mode, 'Application_order': ap_order,
                'Course': course, 'Daytime_evening_attendance': dt_attendance_key,
                'Previous_qualification': pq_key, 'Previous_qualification_grade': pq_grade,
                'Nacionality': nationality, 'Mothers_qualification': mother_qual, 'Fathers_qualification': father_qual,
                'Mothers_occupation': mother_occup, 'Fathers_occupation': father_occup,
                'Admission_grade': adm_grade, 'Displaced': displaced_key,
                'Educational_special_needs': edu_special_key,'Debtor': debtor_key,
                'Tuition_fees_up_to_date': tuition_fees_key,'Gender': gender_key,
                'Scholarship_holder': scholarship_key,'Age_at_enrollment': age_enroll,
                'International': international_key,
                'Curricular_units_1st_sem_credited': cu1_credited,
                'Curricular_units_1st_sem_enrolled': cu1_enrolled,
                'Curricular_units_1st_sem_evaluations': cu1_evals,
                'Curricular_units_1st_sem_approved': cu1_approved,
                'Curricular_units_1st_sem_grade': cu1_grade,
                'Curricular_units_2nd_sem_credited': cu2_credited,
                'Curricular_units_2nd_sem_enrolled': cu2_enrolled,
                'Curricular_units_2nd_sem_evaluations': cu2_evals,
                'Curricular_units_2nd_sem_approved': cu2_approved,
                'Curricular_units_2nd_sem_grade': cu2_grade,
                'pass_ratio_sem1': pass_ratio_sem1_val,
                'pass_ratio_sem2': pass_ratio_sem2_val,
                'total_enrolled': total_enrolled_val,
                'average_grade': average_grade_val,
                'Unemployment_rate': unemployment_rate_val,
                'Inflation_rate': inflation_rate_val,
                'GDP': gdp_val
            }
            
            input_data_dict = {k: [v] for k, v in input_data_dict_temp.items()}
            input_df_all_features = pd.DataFrame(input_data_dict)
            
            expected_features = None
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                missing_from_form = set(expected_features) - set(input_df_all_features.columns)
                if missing_from_form:
                    st.error(f"ERROR KRITIS: Fitur berikut diharapkan oleh model tetapi TIDAK ADA di form/data input: {missing_from_form}. Harap perbarui form di app.py.")
                    st.stop()
                try:
                    input_df = input_df_all_features[expected_features]
                except KeyError as e:
                    st.error(f"KeyError saat mencocokkan fitur: {e}. Fitur yang hilang: {set(expected_features) - set(input_df_all_features.columns)}")
                    st.error(f"Fitur yang tersedia: {input_df_all_features.columns.tolist()}")
                    st.error(f"Fitur yang diharapkan model: {list(expected_features)}")
                    st.stop()
            else:
                st.warning("Atribut `model.feature_names_in_` tidak ditemukan. Menggunakan semua input dari form. Pastikan ini sesuai dengan training model.")
                input_df = input_df_all_features 

            st.markdown("---")
            st.subheader("📊 Data Input Mahasiswa (Dikirim ke Model):")
            st.dataframe(input_df)

            try:
                prediction = model.predict(input_df)
                proba = model.predict_proba(input_df)
                
                predicted_status_val = prediction[0] 
                
                st.subheader("🎯 Hasil Prediksi Utama:")
                
                class_mapping = {0: 'Graduate/Enrolled', 1: 'Dropout'} 
                dropout_class_code = 1 

                predicted_label = class_mapping.get(predicted_status_val, f"Kelas {predicted_status_val} (Tidak Terdefinisi)")
                probability_of_predicted_label = proba[0][predicted_status_val] if proba.shape[1] > predicted_status_val else -1.0

                probability_dropout = -1.0
                if proba.shape[1] > dropout_class_code:
                    probability_dropout = proba[0][dropout_class_code]
                else:
                    st.warning(f"Array probabilitas model (shape: {proba.shape}) tidak memiliki cukup kolom untuk mengambil probabilitas kelas Dropout (indeks {dropout_class_code}).")

                if 'dropout' in predicted_label.lower():
                    st.error(f"🚨 **Prediksi Status: {predicted_label.upper()}**")
                else:
                    st.success(f"✅ **Prediksi Status: {predicted_label.upper()}**")
                
                col_prob1, col_prob2 = st.columns(2)
                if probability_of_predicted_label >=0:
                    with col_prob1:
                        st.metric(label=f"Probabilitas untuk Prediksi ({predicted_label})", value=f"{probability_of_predicted_label*100:.2f}%")
                if probability_dropout >= 0: 
                    with col_prob2:
                         st.metric(label="Probabilitas Menjadi 'Dropout'", value=f"{probability_dropout*100:.2f}%",
                                   help="Ini adalah probabilitas mahasiswa diklasifikasikan sebagai 'Dropout'.")

                with st.expander("🔍 Lihat Penjelasan Detail Prediksi (SHAP Values)", expanded=False):
                    st.markdown("SHAP membantu memahami kontribusi setiap fitur terhadap prediksi.")
                    try:
                        if "LGBMClassifier" in str(type(model)) or "XGBClassifier" in str(type(model)) or "RandomForestClassifier" in str(type(model)) :
                            explainer = shap.TreeExplainer(model)
                            shap_values_from_explainer = explainer.shap_values(input_df) 
                            
                            shap_values_for_class_plot = None 
                            base_value_for_class_plot = None  

                            if isinstance(shap_values_from_explainer, list) and len(shap_values_from_explainer) >= (dropout_class_code + 1):
                                shap_values_for_class_plot = shap_values_from_explainer[dropout_class_code]
                                if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > dropout_class_code:
                                    base_value_for_class_plot = explainer.expected_value[dropout_class_code]
                                elif not isinstance(explainer.expected_value, (list, np.ndarray)): 
                                    base_value_for_class_plot = explainer.expected_value
                                else: 
                                     base_value_for_class_plot = explainer.expected_value[0] if hasattr(explainer.expected_value, '__len__') and len(explainer.expected_value)>0 else 0.0
                                st.info(f"SHAP (dari list): Menampilkan SHAP values untuk kelas target '{class_mapping.get(dropout_class_code)}' (indeks {dropout_class_code}).")
                            
                            elif isinstance(shap_values_from_explainer, np.ndarray):
                                if shap_values_from_explainer.ndim == 2 : 
                                    shap_values_for_class_plot = shap_values_from_explainer
                                    base_value_for_class_plot = explainer.expected_value 
                                    if isinstance(base_value_for_class_plot, (list, np.ndarray)):
                                        base_value_for_class_plot = base_value_for_class_plot[0] if hasattr(base_value_for_class_plot, '__len__') and len(base_value_for_class_plot)>0 else 0.0
                                    st.info("SHAP (dari array 2D): Diasumsikan SHAP values untuk kelas positif (Dropout).")
                                
                                elif shap_values_from_explainer.ndim == 3: 
                                    st.info(f"SHAP (dari array 3D): Bentuk terdeteksi: {shap_values_from_explainer.shape}.")
                                    if shap_values_from_explainer.shape[0] == 1 and shap_values_from_explainer.shape[2] > dropout_class_code:
                                        shap_values_for_class_plot = shap_values_from_explainer[0, :, dropout_class_code] 
                                        
                                        if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > dropout_class_code:
                                            base_value_for_class_plot = explainer.expected_value[dropout_class_code]
                                        elif not isinstance(explainer.expected_value, (list, np.ndarray)): 
                                            base_value_for_class_plot = explainer.expected_value
                                        else: 
                                            base_value_for_class_plot = explainer.expected_value[0] if hasattr(explainer.expected_value, '__len__') and len(explainer.expected_value)>0 else 0.0
                                            st.warning("Panjang expected_value tidak cukup untuk dropout_class_code, menggunakan elemen pertama.")
                                        st.info(f"SHAP (dari array 3D): Dipilih SHAP values untuk kelas '{class_mapping.get(dropout_class_code)}' (indeks {dropout_class_code}).")
                                    else:
                                        st.warning(f"Bentuk array SHAP 3D ({shap_values_from_explainer.shape}) atau indeks kelas target ({dropout_class_code}) tidak sesuai.")
                                else:
                                     st.warning(f"Struktur SHAP values (ndarray) tidak terduga: Dimensi {shap_values_from_explainer.ndim}")
                            else:
                                st.warning(f"Struktur SHAP values tidak terduga: Tipe {type(shap_values_from_explainer)}")


                            if shap_values_for_class_plot is not None and base_value_for_class_plot is not None:
                                current_shap_instance_values = shap_values_for_class_plot[0] if shap_values_for_class_plot.ndim == 2 else shap_values_for_class_plot

                                if hasattr(current_shap_instance_values, 'shape') and len(current_shap_instance_values.shape) == 1 and len(input_df.columns) == len(current_shap_instance_values):
                                    base_value_scalar = base_value_for_class_plot
                                    if isinstance(base_value_for_class_plot, (list, np.ndarray)):
                                        base_value_scalar = base_value_for_class_plot[0] if hasattr(base_value_for_class_plot, '__len__') and len(base_value_for_class_plot) > 0 else 0.0
                                    
                                    st.subheader("Kontribusi Fitur Individual (Waterfall Plot)")
                                    plt.clf() 
                                    fig_waterfall, ax_waterfall_placeholder = plt.subplots(figsize=(10, 8)) 
                                    shap.waterfall_plot(shap.Explanation(values=current_shap_instance_values, 
                                                                          base_values=base_value_scalar, 
                                                                          data=input_df.iloc[0].values, 
                                                                          feature_names=input_df.columns),
                                                        show=False, max_display=15)
                                    # fig_waterfall.tight_layout() # Bisa dipanggil di sini
                                    st.pyplot(fig_waterfall) 
                                    plt.close(fig_waterfall) 

                                    st.subheader("Dorongan Fitur (Force Plot)")
                                    try:
                                        plt.clf() 
                                        fig_force, ax_force_placeholder = plt.subplots(figsize=(12, 3)) # Beri ruang lebih untuk label
                                        shap.force_plot(base_value_scalar,
                                                      current_shap_instance_values,
                                                      input_df.iloc[0],
                                                      matplotlib=True, 
                                                      show=False, 
                                                      # ax=ax_force_placeholder, # Dihapus
                                                      link="logit",
                                                      text_rotation=15 # Coba tambahkan rotasi teks jika label tumpang tindih
                                                     )
                                        fig_force.tight_layout(pad=0.1) # Sesuaikan padding
                                        st.pyplot(fig_force)
                                        plt.close(fig_force)
                                    except Exception as e_force_plot:
                                        st.error(f"Error saat membuat SHAP Force Plot (matplotlib): {e_force_plot}")
                                        st.markdown("Mencoba fallback ke force plot HTML (mungkin memerlukan interaksi browser).")
                                        try: 
                                            shap.initjs()
                                            force_plot_obj = shap.force_plot(base_value_scalar,
                                                                              current_shap_instance_values,
                                                                              input_df.iloc[0],
                                                                              link="logit")
                                            st.components.v1.html(force_plot_obj.html(), height=150, scrolling=True)
                                        except Exception as e_html_force:
                                            st.error(f"Gagal juga membuat force plot HTML: {e_html_force}")

                                else:
                                    st.error(f"Dimensi SHAP values ({current_shap_instance_values.shape if hasattr(current_shap_instance_values, 'shape') else 'N/A'}) atau jumlah fitur tidak cocok untuk waterfall plot.")
                                    st.write("current_shap_instance_values:", current_shap_instance_values)
                                    st.write("base_value_for_class_plot:", base_value_for_class_plot)
                            else:
                                st.error("Tidak dapat menentukan SHAP values atau base value yang sesuai untuk plot setelah memproses struktur output explainer.")
                        else:
                            st.info(f"Tipe model saat ini ({type(model)}) mungkin tidak secara langsung didukung oleh SHAP TreeExplainer. Penjelasan SHAP mungkin terbatas.")
                    
                    except ImportError:
                        st.error("Library SHAP belum terinstal. Silakan instal dengan `pip install shap` untuk melihat penjelasan ini.")
                    except Exception as e_shap:
                        st.error(f"Terjadi error saat membuat penjelasan SHAP: {e_shap}")
                        st.error("Pastikan model Anda kompatibel dengan SHAP dan data input sudah benar.")

            except Exception as e_pred:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {e_pred}")
                st.error("Pastikan semua fitur yang dibutuhkan model Anda telah dimasukkan dengan benar dan formatnya sesuai.")

# --- Sidebar Caption dengan Timestamp ---
st.sidebar.markdown("---")
model_load_time_str = st.session_state.model_load_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(st.session_state.model_load_time, datetime) else "N/A"
data_viz_load_time_str = st.session_state.data_viz_load_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(st.session_state.data_viz_load_time, datetime) else "N/A"

st.sidebar.caption(f"""
Versi Aplikasi: {APP_VERSION}<br>
Status Model: {st.session_state.model_status}<br>
Model Dimuat: {model_load_time_str}<br>
Data Visualisasi Dimuat: {data_viz_load_time_str}
""", unsafe_allow_html=True)