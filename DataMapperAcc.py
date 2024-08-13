import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import matplotlib.pyplot as plt

# Custom CSS to ensure full-screen layout and minimal margins
st.markdown("""
    <style>
        .title { font-size: 36px; font-weight: bold; color: #4CAF50; }
        .header { font-size: 24px; font-weight: bold; color: #2196F3; }
        .stApp { margin: 0; padding: 0; }
        .stButton>button { padding: 10px 20px; }
        .stTabs>div>div>button { margin: 0; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Data Mapper</h1>", unsafe_allow_html=True)

# Define tabs
tabs = st.tabs(["Upload Files", "Data Overview", "Matching Results", "Unmatched Data", "Download Results", "Validation"])

# Define global variables
df1 = None
df2 = None
fuzzy_matches = None

with tabs[0]:
    st.markdown("## Upload Your Files")
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file1 = st.file_uploader("Choose the first CSV file (df1)", type="csv")
    with col2:
        uploaded_file2 = st.file_uploader("Choose the second CSV file (df2)", type="csv")

    with st.form(key='upload_form'):
        uploaded_val = st.file_uploader("Choose the validation CSV file (val)", type="csv")
        submit_button = st.form_submit_button(label="Upload and Process")

with tabs[1]:
    if uploaded_file1 is not None and uploaded_file2 is not None and submit_button:
        st.spinner('Processing...')
        df1 = pd.read_csv(uploaded_file1)
        df2 = pd.read_csv(uploaded_file2)

        df1 = df1.rename(columns={'INSTNM': 'inst', 'ID': 'id', 'STABBR': 'state', 'CITY': 'city'})
        df2 = df2.rename(columns={'Institution Name': 'inst', 'State': 'state'})
        
        df1 = df1.replace('--', np.nan).dropna().drop_duplicates()
        df2 = df2.replace('--', np.nan).dropna().drop_duplicates()
        
        df1['inst'] = df1['inst'].str.lower()
        df2['inst'] = df2['inst'].str.lower()
        
        st.markdown("## Data Overview")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("### Dataframe 1:")
            st.write(df1.head())
        with col2:
            st.write("### Dataframe 2:")
            st.write(df2.head())

        st.write(f"**df1 shape**: {df1.shape}")
        st.write(f"**df2 shape**: {df2.shape}")

with tabs[2]:
    if df1 is not None and df2 is not None and submit_button:
        st.markdown("## Matching Results")
        threshold = st.slider('Similarity Threshold', 50, 100, 70)
        
        with st.spinner('Performing fuzzy matching...'):
            matches = []
            for index, row in df2.iterrows():
                match = process.extractOne(row['inst'], df1['inst'], scorer=fuzz.token_sort_ratio)
                if match[1] > threshold:
                    df1_matched_row = df1[df1['inst'] == match[0]]
                    if df1_matched_row['state'].values[0] == row['state']:
                        matches.append({
                            'df2_inst': row['inst'],
                            'df2_state': row['state'],
                            'df1_inst': df1_matched_row['inst'].values[0],
                            'df1_state': df1_matched_row['state'].values[0],
                            'similarity': match[1]
                        })

            fuzzy_matches = pd.DataFrame(matches)
        
        st.write(f"**Fuzzy Matches ({fuzzy_matches.shape[0]})**")
        st.write(fuzzy_matches.head())
        
        st.write("### Similarity Scores Distribution")
        fig, ax = plt.subplots()
        fuzzy_matches['similarity'].hist(ax=ax)
        st.pyplot(fig)

with tabs[3]:
    if df1 is not None and df2 is not None and fuzzy_matches is not None:
        st.markdown("## Unmatched Data")
        matched_insts = fuzzy_matches['df2_inst']
        unmatched_rows = df2[~df2['inst'].isin(matched_insts)]
        st.write(f"**Unmatched Rows ({unmatched_rows.shape[0]})**")
        st.write(unmatched_rows)
        
        st.download_button(
            label="Download Unmatched Rows",
            data=unmatched_rows.to_csv(index=False).encode("utf-8"),
            file_name="unmatched_rows.csv",
            mime="text/csv"
        )

with tabs[4]:
    if fuzzy_matches is not None:
        st.markdown("## Download Results")
        
        st.download_button(
            label="Download Fuzzy Matches",
            data=fuzzy_matches.to_csv(index=False).encode("utf-8"),
            file_name="fuzzy_matched.csv",
            mime="text/csv"
        )
        
        remaining_matches = pd.merge(df1, unmatched_rows, on=['inst', 'state'])
        st.download_button(
            label="Download Remaining Matches",
            data=remaining_matches.to_csv(index=False).encode("utf-8"),
            file_name="remaining_matches.csv",
            mime="text/csv"
        )

with tabs[5]:
    if uploaded_val is not None and fuzzy_matches is not None:
        st.markdown("## Validation Results")
        
        df_val = pd.read_csv(uploaded_val)
        
        df_val['df1_inst'] = df_val['df1_inst'].str.lower()
        df_val['df2_inst'] = df_val['df2_inst'].str.lower()
        
        # Step 1: Note the shape of validation.csv before merging
        initial_shape_val = df_val.shape[0]
        
        # Merge the two DataFrames on all relevant columns
        merged_df = df_val.merge(fuzzy_matches, on=['df1_inst', 'df1_state', 'df2_inst', 'df2_state'], how='left', indicator=True)
        
        # Step 2: Remove the merged rows from validation.csv
        df_val_remaining = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        
        # Step 3: Note the shape of validation.csv after removal
        remaining_shape_val = df_val_remaining.shape[0]
        
        # Calculate the number of matched rows and the percentage
        matched_rows = initial_shape_val - remaining_shape_val
        percentage_matched = (matched_rows / initial_shape_val) * 100
        
        st.write(f"**Initial number of rows in val.csv**: {initial_shape_val}")
        st.write(f"**Number of matched rows**: {matched_rows}")
        st.write(f"**Percentage of rows matched**: {percentage_matched:.2f}%")
        
        st.download_button(
            label="Download Validation Results",
            data=df_val_remaining.to_csv(index=False).encode("utf-8"),
            file_name="validation_remaining.csv",
            mime="text/csv"
        )
