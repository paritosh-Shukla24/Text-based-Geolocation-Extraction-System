import textwrap
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import faiss
import spacy
from geopy.geocoders import Nominatim
import folium
from folium.plugins import MarkerCluster

# Initialize global variables for ML model
documents = None
db = None
nlp = None

def process_query(query):
    from langchain_community.document_loaders import TextLoader
    loader=TextLoader('data.txt',encoding='utf-8')
    documents=loader.load()

    import textwrap

    from langchain.text_splitter import CharacterTextSplitter
    text_splitter=CharacterTextSplitter(chunk_size=10,chunk_overlap=0)  
    docs=text_splitter.split_documents(documents)

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import faiss
    embeddings=HuggingFaceEmbeddings()
    db=faiss.FAISS.from_documents(docs,embeddings)
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk.download('stopwords')
    nltk.download('punkt')
    word_tokens = word_tokenize(query)
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence=' '.join(filtered_sentence)
    doc=db.similarity_search(filtered_sentence)
    a=wrap_text_preserve_newline(str(doc[0].page_content))
    a=a.split()
    text=' '.join(a)
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    lat,long=get_coordinates(locations[0])
    map_centered=open_map_with_marker(lat,long)
    return map_centered


def wrap_text_preserve_newline(text, width=10):
    lines = text.split()
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text


def get_coordinates(location):
    geolocator = Nominatim(user_agent="my_geocoding_app")
    location_data = geolocator.geocode(location)
    if location_data:
        latitude, longitude = location_data.latitude, location_data.longitude
        return latitude, longitude
    else:
        return None


def open_map_with_marker(latitude, longitude, zoom_level=10):
    map_centered = folium.Map(location=[latitude, longitude], zoom_start=zoom_level)
    marker_cluster = MarkerCluster().add_to(map_centered)
    darker_marker = folium.Marker(
        location=[latitude, longitude],
        icon=folium.Icon(color='darkpurple', icon_color='white', icon='circle', prefix='fa')
    ).add_to(marker_cluster)

    # Save the map as map.html
    return map_centered
