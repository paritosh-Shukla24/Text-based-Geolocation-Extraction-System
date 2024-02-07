from transformers import AutoTokenizer, AutoModelForSequenceClassification
import folium
from folium.plugins import MarkerCluster

import torch
model_name = "jb2k/bert-base-multilingual-cased-language-detection"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)



text='مرحبا بالعالم'
encoded_text = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**encoded_text)
    logits = outputs.logits.squeeze(0)  

predicted_language = torch.argmax(logits).item()  
language_id_to_name = {
    0: "Arabic", 1: "Basque", 2: "Breton", 3: "Catalan", 4: "Chinese (China)", 5: "Chinese (Hong Kong)", 6: "Chinese (Taiwan)", 7: "Chuvash", 8: "Czech", 9: "Dhivehi", 10: "Dutch",
    11: "English", 12: "Esperanto", 13: "Estonian", 14: "French", 15: "Frisian", 16: "Georgian", 17: "German", 18: "Greek", 19: "Hakha Chin", 20: "Indonesian", 21: "Interlingua",
    22: "Italian", 23: "Japanese", 24: "Kabyle", 25: "Kinyarwanda", 26: "Kyrgyz", 27: "Latvian", 28: "Maltese", 29: "Mongolian", 30: "Persian", 31: "Polish", 32: "Portuguese",
    33: "Romanian", 34: "Romansh Sursilvan", 35: "Russian", 36: "Sakha", 37: "Slovenian", 38: "Spanish", 39: "Swedish", 40: "Tamil", 41: "Tatar", 42: "Turkish", 43: "Ukrainian", 44: "Welsh"
}
predicted_language_name = language_id_to_name[predicted_language]

print("Predicted language:", predicted_language_name)


language_to_country = {
    "Arabic": "Arab League",
    "Basque": "Basque Country",
    "Breton": "France",
    "Catalan": "Spain, France, Italy",
    "Chinese (China)": "China",
    "Chinese (Hong Kong)": "Hong Kong",
    "Chinese (Taiwan)": "Taiwan",
    "Chuvash": "Chuvash Republic, Russia",
    "Czech": "Czech Republic",
    "Dhivehi": "Maldives",
    "Dutch": "Netherlands, Belgium",
    "English": "United Kingdom, United States, Australia, New Zealand, Canada, and others",
    "Esperanto": "Constructed language, no specific country",
    "Estonian": "Estonia",
    "French": "France, Canada, Belgium, Switzerland, and others",
    "Frisian": "Netherlands, Germany",
    "Georgian": "Georgia",
    "German": "Germany, Austria, Switzerland, Belgium, Luxembourg, and others",
    "Greek": "Greece, Cyprus",
    "Hakha Chin": "Myanmar, India, Bangladesh",
    "Indonesian": "Indonesia",
    "Interlingua": "Constructed language, no specific country",
    "Italian": "Italy, Switzerland, San Marino, Vatican City",
    "Japanese": "Japan",
    "Kabyle": "Algeria",
    "Kinyarwanda": "Rwanda, Uganda, Democratic Republic of the Congo",
    "Kyrgyz": "Kyrgyzstan",
    "Latvian": "Latvia",
    "Maltese": "Malta",
    "Mongolian": "Mongolia",
    "Persian": "Iran, Afghanistan, Tajikistan",
    "Polish": "Poland",
    "Portuguese": "Portugal, Brazil, Mozambique, Angola, and others",
    "Romanian": "Romania, Moldova",
    "Romansh Sursilvan": "Switzerland",
    "Russian": "Russia, Belarus, Kyrgyzstan, Kazakhstan, and others",
    "Sakha": "Sakha Republic, Russia",
    "Slovenian": "Slovenia",
    "Spanish": "Spain, Latin America, Equatorial Guinea",
    "Swedish": "Sweden, Finland",
    "Tamil": "India, Sri Lanka, Singapore, Malaysia",
    "Tatar": "Tatarstan, Russia",
    "Turkish": "Turkey, Cyprus",
    "Ukrainian": "Ukraine",
    "Welsh": "Wales, United Kingdom"
}




predicted_country = language_to_country[predicted_language_name]




from geopy.geocoders import Nominatim

def get_coordinates(location):
    # Set your app name or any string to identify your application
    geolocator = Nominatim(user_agent="my_geocoding_app",timeout=10)
    location_data = geolocator.geocode(location)

    if location_data:
        latitude, longitude = location_data.latitude, location_data.longitude
        return latitude, longitude
    else:
        return None




location = predicted_country
coordinates = get_coordinates(location)




lat,long=coordinates


def open_map_with_marker(latitude, longitude, zoom_level=10):
    map_centered = folium.Map(location=[latitude, longitude], zoom_start=zoom_level)
    marker_cluster = MarkerCluster().add_to(map_centered)
    darker_marker = folium.Marker(
        location=[latitude, longitude],
        icon=folium.Icon(color='darkpurple', icon_color='white', icon='circle', prefix='fa')
    ).add_to(marker_cluster)

    # Save the map as map.html
    return map_centered


open_map_with_marker(lat,long)