
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util




def similarity_document(df):
    docs = df['AB'].tolist()

    # Load a pre-trained biomedical model
    biomedical_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Sample biomedical text data
    biomedical_texts = docs

    # Generate embeddings for biomedical texts
    biomedical_embeddings = biomedical_model.encode(biomedical_texts)

    # Streamlit app layout
    st.title("Biomedical Text Similarity App")

    # Sidebar with query text box
    query = st.sidebar.text_input("Enter your query:")
    if not query:
        st.sidebar.warning("Please enter a query.")
        st.stop()

    # Calculate similarity scores
    query_embedding = biomedical_model.encode([query])
    similarity_scores = util.pytorch_cos_sim(query_embedding, biomedical_embeddings).flatten()



    #doc_score_pairs = list(zip(biomedical_texts, similarity_scores))
    df["scores"] = similarity_scores
    df_sorted = df.sort_values(by='scores', ascending = False)
    df_sorted = df_sorted[["AB", "PMID", "scores"]][:5]
    #doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)



    # Display text and similarity scores
    st.subheader("Top Similar Texts:")
    for row in df_sorted.itertuples(index=False):
        st.write(f"Text: {row.AB}")
        st.write(f"   Similarity Score: {row.scores:.4f}")
        pmid = row.PMID
        st.write(f"   PMID: {pmid}")
        pubmed_url = f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'
        st.markdown(f'PubMed URL: [{pubmed_url}]({pubmed_url})')
        st.markdown("---")  # Separator






def plot_top_journals(df):
    top_journals = df['JT'].value_counts().index[:5].tolist()

    st.write("<h3 style='color: #339af0;'>Top 5 Journals by Number of Records:</h3>", unsafe_allow_html=True)
    
    for i, journal in enumerate(top_journals, start=1):
        st.write(f"<span style='font-size: 16px;'><b>{i}. {journal}</b></span>", unsafe_allow_html=True)


def plot_publications_per_year(df):
    df['PubDate'] = pd.to_datetime(df['DP'], errors='coerce')
    df['Year'] = df['PubDate'].dt.year
    df['Month'] = df['PubDate'].dt.month
    publications_per_year = df['Year'].value_counts().sort_index()

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=publications_per_year.index, y=publications_per_year.values, marker='o', color='skyblue')
    plt.title('Number of Publications per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Publications')
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(plt)

def plot_publications_per_month(df):
    last_24_months = datetime.now() - timedelta(days=24 * 30)
    df['PubDate'] = pd.to_datetime(df['DP'], errors='coerce')
    df_last_24_months = df[df['PubDate'] >= last_24_months]
    df_last_24_months['YearMonth'] = df_last_24_months['PubDate'].dt.to_period('M')
    publications_per_month = df_last_24_months['YearMonth'].value_counts().sort_index()

    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 6))
    sns.lineplot(x=publications_per_month.index.astype(str), y=publications_per_month.values, marker='o',
                 color='skyblue')
    plt.title('Number of Publications per Month (Last 24 Months)')
    plt.xlabel('Year-Month')
    plt.ylabel('Number of Publications')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(plt)


def main():
    st.title("PubMed Data Analysis")
    diseases = [
    "Cardiovascular Diseases",
    "Cancer",
    "Respiratory Diseases",
    "Infectious Diseases",
    "Alzheimer's",
    "Obesity",
    "Mental Health Disorders",
    "Dengue",
    "COVID-19",
    "Diabetes",
    ]

# Create a dropdown menu
    selected_disease = st.sidebar.selectbox("Select a disease", diseases)
    df = pd.read_csv(f"./{selected_disease.replace(' ','')}.csv")

    df.dropna(subset=['AB'], inplace=True)
    df.reset_index(drop=True,inplace=True)  

    # Sidebar for user selection
    #analysis_option = st.sidebar.selectbox("Select Analysis", ["Top Authors", "Publications per Year","Publications per Month"])

    #if analysis_option == "Top Authors":
    plot_top_journals(df)
    #elif analysis_option == "Publications per Year":
    plot_publications_per_year(df)
    #elif analysis_option == "Publications per Month":
    plot_publications_per_month(df)
    similarity_document(df)

if __name__ == "__main__":
    main()
