import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt

join = []
temp = []

with st.sidebar:
  selected = option_menu('Ekstraksi Kata', ['Load Data', 'Preprocessing', 'Matriks Koherensi', 'Graph', 'Closeness Centrality', 'Pagerank', 'Eignvector Centrality', 'Betweeness Centrality'], default_index=0)
st.title("Ekstraksi Kata")
##### Crwaling Data
# def crawlingPta():
#   st.subheader("Crawling Portal Media Indonesia")
#   url = st.text_input('Inputkan url mediaindonesi berdasarkan topik di sini', 'https://mediaindonesia.com/politik-dan-hukum')
#   button = st.button('Crawling')
#   if (button):
#     # daftar fungsi yang dibutuhkan
#     def request_url(url):
#       return requests.get(url)

#     def request_header_url(header, url_website):
#         headers = {'User-Agent': header}
#         return requests.get(url=url_website, headers=headers)

#     def parse_website(request):
#         """Use html5lib library to parse"""
#         return BeautifulSoup(request.content, 'html5lib') # If this line causes an error, run 'pip install html5lib' or install html5lib    

#     def prettify_web_structure(parsed_page):
#         parsed_page.prettify()

#     def get_content_table(web_element, tag, attributes):
#         return web_element.find(tag, attrs = attributes)

#     r = request_header_url("Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1", url)
#     soup = parse_website(r)
#     prettify_web_structure(soup)
#     table = get_content_table(soup, "ul", {'class':'article-block'})

#     articles = []
#     for row in table.findAll('a', attrs = {'class':'hover-effect'}):
#       current_title = row["title"]

#       r_row = request_header_url("Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1", row["href"])
#       soup_row = parse_website(r_row)
#       prettify_web_structure(soup_row)
#       table_row = get_content_table(soup_row, "div", {"style":"line-height: 1.6;", "itemprop":"articleBody"})

#       current_content = ""
#       for inner_row in table_row.findAll('p'):
#         current_content += f" {inner_row.string}"
#       # dump variable 
#       articles.append([current_title, current_content])
#     # data = {'Judul': articles[0], 'Isi': articles[1]}
#     dfData = pd.DataFrame(articles, columns=["Judul", "Isi"])
#     st.dataframe(dfData)

##### Load Data
def loadData():
  st.subheader("Load Data:")
  data_url = st.text_input('Enter URL of your CSV file here', 'https://raw.githubusercontent.com/dennywr/cobaprosaindata/main/kalimat.csv')

  @st.cache_resource
  def load_data():
      data = pd.read_csv(data_url, index_col=False)
      # data['nomor\ufeff'] += 1
      return data

  df = load_data()
  # df.set_index('nomor\ufeff', inplace=True)
  # df.index += 1
  df['isi'] = df['isi'].fillna('').astype(str)
  # if(selected == 'Load Data'):
  st.dataframe(df)
  return (df['isi'])
    

##### Preprocessing
def preprocessing():
  st.subheader("Preprocessing:")
  st.text("Menghapus karakter spesial")

  ### hapus karakter spesial
  @st.cache_resource
  def load_data():
      data = pd.read_csv('https://raw.githubusercontent.com/dennywr/cobaprosaindata/main/kalimat.csv', index_col=False)
      # data['nomor\ufeff'] += 1
      return data

  df = load_data()
  import nltk
  from nltk.corpus import stopwords
  nltk.download('stopwords')

  def removeSpecialText (text):
    text = text.replace('\\t',"").replace('\\n',"").replace('\\u',"").replace('\\',"").replace('None',"")
    text = text.encode('ascii', 'replace').decode('ascii')
    return text.replace("http://"," ").replace("https://", " ")
  
  df['isi'] = df['isi'].astype(str).apply(removeSpecialText)
  df['isi'] = df['isi'].apply(removeSpecialText)
  # df.index += 1
  st.dataframe(df['isi'])

  ### hapus tanda baca
  st.text("Menghapus tanda baca")
  def removePunctuation(text):
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text)
    return text

  df['isi'] = df['isi'].apply(removePunctuation)
  st.dataframe(df['isi'])

  ### hapus angka pada teks
  st.text("Menghapus angka pada teks")
  def removeNumbers (text):
    return re.sub(r"\d+", "", text)
  df['isi'] = df['isi'].apply(removeNumbers)
  st.dataframe(df['isi'])

  ### case folding
  st.text("Mengubah semua huruf pada teks menjadi huruf kecil")
  def casefolding(Comment):
    Comment = Comment.lower()
    return Comment
  df['isi'] = df['isi'].apply(casefolding)
  st.dataframe(df['isi'])

  ### stopwords removal
  st.text("Melakukan penghapusan stopwords")
  def removeStopwords(text):
    stop_words = set(stopwords.words('indonesian'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

  df['isi'] = df['isi'].apply(removeStopwords)
  st.dataframe(df['isi'])
  
  #   dfRemoved = pd.DataFrame(removed, columns=['Tokenisasi dan Stopwords']).T
  # # Display the DataFrame
  # st.dataframe(dfRemoved.head(5))


def preprocessingTanpaOutput():

  ### hapus karakter spesial
  @st.cache_resource
  def load_data():
      data = pd.read_csv('https://raw.githubusercontent.com/dennywr/cobaprosaindata/main/kalimat.csv')
      return data

  df = load_data()
  ### if(hapusKarakterSpesial):
  import nltk
  from nltk.corpus import stopwords
  nltk.download('stopwords')
  def removeSpecialText (text):
    text = text.replace('\\t',"").replace('\\n',"").replace('\\u',"").replace('\\',"").replace('None',"")
    text = text.encode('ascii', 'replace').decode('ascii')
    return text.replace("http://"," ").replace("https://", " ")
  
  df['isi'] = df['isi'].astype(str).apply(removeSpecialText)
  df['isi'] = df['isi'].apply(removeSpecialText)


  # hapusTandaBaca = st.button("Hapus Tanda Baca")
  def removePunctuation(text):
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text)
    return text

  df['isi'] = df['isi'].apply(removePunctuation)

  ### hapus angka pada teks
  def removeNumbers (text):
    return re.sub(r"\d+", "", text)
  df['isi'] = df['isi'].apply(removeNumbers)

  ### case folding
  def casefolding(Comment):
    Comment = Comment.lower()
    return Comment
  df['isi'] = df['isi'].apply(casefolding)

  ### stopwords removal
  def removeStopwords(text):
    stop_words = set(stopwords.words('indonesian'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)
  df['isi'] = df['isi'].apply(removeStopwords)

  return (df["isi"])


import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from itertools import combinations
import numpy as np
def build_cooccurrence_matrix():
  def cooccurrence_matrix(sentences, window_size):
    # Inisialisasi dictionary untuk menyimpan matriks koherensi antara kata-kata
    cooccurrence_dict = {}

    # Loop untuk setiap kalimat dalam daftar kalimat
    for sentence in sentences:
        # Tokenisasi kata-kata dalam kalimat dan ubah ke huruf kecil
        words = word_tokenize(sentence.lower())

        # Loop untuk setiap kata dalam kalimat bersama dengan indeksnya
        for i, word1 in enumerate(words):
            # Tentukan jangkauan indeks sebagai "window" di sekitar kata saat ini
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)

            # Buat daftar kata-kata dalam window dan hapus kata saat ini dari konteks
            context_words = words[start:end]
            context_words.remove(word1)

            # Loop untuk setiap kata dalam konteks
            for word2 in context_words:
                # Inisialisasi array nol jika kata belum ada dalam dictionary
                if word1 not in cooccurrence_dict:
                    cooccurrence_dict[word1] = {}

                # Tingkatkan nilai dalam matriks koherensi untuk pasangan kata word1 dan word2
                if word2 in cooccurrence_dict[word1]:
                    cooccurrence_dict[word1][word2] += 1
                else:
                    cooccurrence_dict[word1][word2] = 1

    return cooccurrence_dict

  # Ukuran jendela (window size)
  window_size = 1

  # Mengumpulkan semua kalimat dari kolom 'kalimat' dalam dataframe
  kalimat = preprocessingTanpaOutput()

  # Membuat vocabulary dari kata-kata unik dalam kalimat
  vocabulary = list(set(word_tokenize(' '.join(kalimat).lower())))
  st.subheader("Matriks Koherensi:")
  # Membangun matriks koherensi dengan ukuran window
  cooccurrence_matrix_result = cooccurrence_matrix(kalimat, window_size)
  # st.text(cooccurrence_matrix_result)
  # DataFrame dari hasil matriks koherensi
  cooccurrence_matrix_result_df = pd.DataFrame(cooccurrence_matrix_result, index=cooccurrence_matrix_result)
  cooccurrence_matrix_result_df = cooccurrence_matrix_result_df.fillna(0).astype(int)

  st.dataframe(cooccurrence_matrix_result_df)

## tanpa output  
def build_cooccurrence_matrix_tanpa_output():
  def cooccurrence_matrix(sentences, window_size):
    # Inisialisasi dictionary untuk menyimpan matriks koherensi antara kata-kata
    cooccurrence_dict = {}

    # Loop untuk setiap kalimat dalam daftar kalimat
    for sentence in sentences:
        # Tokenisasi kata-kata dalam kalimat dan ubah ke huruf kecil
        words = word_tokenize(sentence.lower())

        # Loop untuk setiap kata dalam kalimat bersama dengan indeksnya
        for i, word1 in enumerate(words):
            # Tentukan jangkauan indeks sebagai "window" di sekitar kata saat ini
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)

            # Buat daftar kata-kata dalam window dan hapus kata saat ini dari konteks
            context_words = words[start:end]
            context_words.remove(word1)

            # Loop untuk setiap kata dalam konteks
            for word2 in context_words:
                # Inisialisasi array nol jika kata belum ada dalam dictionary
                if word1 not in cooccurrence_dict:
                    cooccurrence_dict[word1] = {}

                # Tingkatkan nilai dalam matriks koherensi untuk pasangan kata word1 dan word2
                if word2 in cooccurrence_dict[word1]:
                    cooccurrence_dict[word1][word2] += 1
                else:
                    cooccurrence_dict[word1][word2] = 1

    return cooccurrence_dict

  # Ukuran jendela (window size)
  window_size = 1

  # Mengumpulkan semua kalimat dari kolom 'kalimat' dalam dataframe
  kalimat = preprocessingTanpaOutput()

  # Membuat vocabulary dari kata-kata unik dalam kalimat
  vocabulary = list(set(word_tokenize(' '.join(kalimat).lower())))
  # Membangun matriks koherensi dengan ukuran window
  cooccurrence_matrix_result = cooccurrence_matrix(kalimat, window_size)
  # st.text(cooccurrence_matrix_result)
  # DataFrame dari hasil matriks koherensi
  cooccurrence_matrix_result_df = pd.DataFrame(cooccurrence_matrix_result, index=cooccurrence_matrix_result)
  cooccurrence_matrix_result_df = cooccurrence_matrix_result_df.fillna(0).astype(int)

  return (cooccurrence_matrix_result_df, vocabulary)


def graph():
    G = nx.Graph()

    cooccurrence_matrix, terms = build_cooccurrence_matrix_tanpa_output()
    num_nodes = len(terms)

    # Adding nodes to the graph with term labels
    for i, term in enumerate(terms):
        G.add_node(i, label=term)

    # Adding edges based on cosine similarity
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if cooccurrence_matrix.iloc[i, j] > 0.05:
                G.add_edge(i, j, weight=cooccurrence_matrix.iloc[i, j])

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    
    # Getting labels from the 'label' attribute of nodes
    labels = {i: G.nodes[i]['label'] for i in G.nodes}

    nx.draw(G, pos, with_labels=True, labels=labels, font_size=8, node_size=500, font_color='black')
    st.subheader("Graph:")
    st.pyplot(plt)

def graphTanpaOutput():
  # Membuat grafik kosong

  G = nx.Graph()

  cooccurrence_matrix, terms = build_cooccurrence_matrix_tanpa_output()
  num_nodes = len(terms)

  # Adding nodes to the graph with term labels
  for i, term in enumerate(terms):
      G.add_node(i, label=term)

  # Adding edges based on cosine similarity
  for i in range(num_nodes):
      for j in range(i + 1, num_nodes):
          if cooccurrence_matrix.iloc[i, j] > 0.05:
              G.add_edge(i, j, weight=cooccurrence_matrix.iloc[i, j])

  return G

def closenessCentrality():
  # Menghitung closeness centrality
  G = graphTanpaOutput()
  closeness_centrality = nx.closeness_centrality(G)

  # Membangun kamus yang mengaitkan angka node dengan kata-kata unik
  node_to_term = {i: term for i, term in enumerate(build_cooccurrence_matrix_tanpa_output()[1])}

  st.subheader("Closeness Centrality:")
  # Mencetak hasil
  for node, closeness in closeness_centrality.items():
      term = node_to_term[node]
      st.text(f"Node {term}: Closeness Centrality = {closeness}")

  st.subheader("3 node dengan closeness centrality tertinggi:")
  # Mengurutkan node berdasarkan closeness centrality
  sorted_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)

  # Tampilkan 3 kata teratas dari nilai tertinggi
  top_3_nodes = sorted_closeness[:3]

  results = []
  for rank, (node, closeness) in enumerate(top_3_nodes, start=1):
      term = node_to_term[node]
      results.append((term, closeness))

  df = pd.DataFrame(results, columns=['Node', 'Closeness Centrality'])
  df.index += 1
  st.dataframe(df)

def pagerank():
  # Assuming graphTanpaOutput() returns a NetworkX graph
  G = graphTanpaOutput()

  # Calculate PageRank
  pagerank_values = nx.pagerank(G, alpha=0.85)

  # Assuming node_to_term is a mapping from node indices to terms
  node_to_term = {i: term for i, term in enumerate(build_cooccurrence_matrix_tanpa_output()[1])}

  # Display PageRank values for each node
  st.subheader("Pagerank:")
  for node, rank in pagerank_values.items():
      term = node_to_term[node]
      st.text(f"Node {term}: PageRank = {rank}")

  # Display top 3 nodes with the highest PageRank
  st.subheader("3 node dengan nilai pagerank tertinggi:")
  sorted_pagerank = sorted(pagerank_values.items(), key=lambda x: x[1], reverse=True)

  top_3_nodes = sorted_pagerank[:3]

  results = []
  for rank, (node, pagerank) in enumerate(top_3_nodes, start=1):
      term = node_to_term[node]
      results.append((term, pagerank))

  df = pd.DataFrame(results, columns=['Node', 'Pagerank'])
  df.index += 1
  st.dataframe(df)

def eignvectorCentrality():
  # Menghitung eigenvector centrality
  G = graphTanpaOutput()
  eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=101)

  # Assuming node_to_term is a mapping from node indices to terms
  node_to_term = {i: term for i, term in enumerate(build_cooccurrence_matrix_tanpa_output()[1])}

  st.subheader("Eigenvector Centrality:")

  # Mencetak hasil
  for node, centrality in eigenvector_centrality.items():
      term = node_to_term[node]
      st.text(f"Node {term}: Eigenvector Centrality = {centrality}")

  # Mengurutkan eigenvector centrality dari yang tertinggi ke terendah
  st.subheader("3 node dengan nilai eigenvector centrality tertinggi:")
  sorted_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)

  # Tampilkan 3 kata teratas dari nilai tertinggi
  top_3_nodes = sorted_eigenvector[:3]

  results = []
  for rank, (node, eigenvector) in enumerate(top_3_nodes, start=1):
      term = node_to_term[node]
      results.append((term, eigenvector))

  df = pd.DataFrame(results, columns=['Node', 'Eigenvector Centrality'])
  df.index += 1
  st.dataframe(df)

def betweennessCentrality():
  # Menghitung betweenness centrality
  G = graphTanpaOutput()
  betweenness_centrality = nx.betweenness_centrality(G)

  # Assuming node_to_term is a mapping from node indices to terms
  node_to_term = {i: term for i, term in enumerate(build_cooccurrence_matrix_tanpa_output()[1])}

  st.subheader("Betweenness Centrality:")

  # Mencetak hasil
  for node, centrality in betweenness_centrality.items():
      term = node_to_term[node]
      st.text(f"Node {term}: Betweenness Centrality = {centrality}")

  # Mengurutkan betweenness centrality dari yang tertinggi ke terendah
  st.subheader("3 node dengan nilai betweenness centrality tertinggi:")
  sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)

  # Tampilkan 3 kata teratas dari nilai tertinggi
  top_3_nodes = sorted_betweenness[:3]

  results = []
  for rank, (node, betweenness) in enumerate(top_3_nodes, start=1):
      term = node_to_term[node]
      results.append((term, betweenness))

  df = pd.DataFrame(results, columns=['Node', 'Betweenness Centrality'])
  df.index += 1
  st.dataframe(df)
  

def main():
  # if(selected == 'Crawling Data'):
  #    crawlingPta()

  if(selected == 'Load Data'):
     loadData()
  if(selected == 'Preprocessing'):
     preprocessing()

  if(selected == 'Matriks Koherensi'):
    #  preprocessingOutputHidden()
    build_cooccurrence_matrix()

  if(selected == 'Graph'):
     graph()
  if(selected == 'Closeness Centrality'):
     closenessCentrality()
  if(selected == 'Pagerank'):
     pagerank()
  if(selected == 'Eignvector Centrality'):
     eignvectorCentrality()
  if(selected == 'Betweeness Centrality'):
     betweennessCentrality()




if __name__ == "__main__":
    main()

