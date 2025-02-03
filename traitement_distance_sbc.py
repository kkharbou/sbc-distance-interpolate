import pandas as pd
import streamlit as st
import logging
import math
from math import cos, sin, atan2, radians, degrees, asin
from numpy import sin, cos, arcsin, radians, sqrt
import numpy as np
import pydeck as pdk
from streamlit_folium import st_folium
from folium.features import CustomIcon
from pathlib import Path
import warnings
import pyproj


# Ignorer les avertissements FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

lambert93 = pyproj.CRS("EPSG:2154")  # Lambert 93
wgs84 = pyproj.CRS("EPSG:4326")  # WGS84 (GPS)

# Créer un transformateur entre les systèmes de projection
transformer = pyproj.Transformer.from_crs(lambert93, wgs84, always_xy=True)

# Fonction de conversion des coordonnées Lambert 93 vers WGS84
def lambert93_to_wgs84(x, y):
    lon, lat = transformer.transform(x, y)
    return lon, lat

def calculate_bearing(lat1, lng1, lat2, lng2):
    """
    A function to calculate azimuth
    """
 
    start_lat = math.radians(lat1)
    start_long = math.radians(lng1)
    end_lat = math.radians(lat2)
    end_long = math.radians(lng2)
    d_long = end_long - start_long
    d_phi = math.log(math.tan(end_lat / 2.0 + math.pi / 4.0) / math.tan(start_lat / 2.0 + math.pi / 4.0))
    if abs(d_long) > math.pi:
        if d_long > 0.0:
            d_long = -(2.0 * math.pi - d_long)
        else:
            d_long = (2.0 * math.pi + d_long)
    bearing = (math.degrees(math.atan2(d_long, d_phi)) + 360.0) % 360.0
 
    return bearing


def get_destination_lat_long(lat, lng, azimuth, distance):
    """
    A function to calculate coordinates
    """
 
    # azimuth, lat, lng: array of 1 element ==> we use .iloc[-1] to get this element
 
    radius = 6373 #Radius of the Earth in km
    brng = radians(azimuth) #Bearing is degrees converted to radians.
    dist = (distance)/1000 #Distance m converted to km
 
    lat1 = radians(lat) #Current dd lat point converted to radians
    lon1 = radians(lng) #Current dd long point converted to radians
 
    lat2 = asin(sin(lat1) * cos(dist/radius) + cos(lat1)* sin(dist/radius)* cos(brng))
 
    lon2 = lon1 + atan2(sin(brng) * sin(dist/radius)* cos(lat1), cos(dist/radius)- sin(lat1)* sin(lat2))
 
    #convert back to degrees
    lat2 = degrees(lat2)
    lon2 = degrees(lon2)
 
    return[lat2, lon2]

 
def approx_flying_distance_in_m(ref_latitude, ref_longitude, points_latitude, points_longitude):
    """
    https://en.wikipedia.org/wiki/Great-circle_distance
 
    :param ref_latitude:
    :param ref_longitude:
    :param points_latitude:
    :param points_longitude:
 
    :return:
 
    """
    # earth radius
    radius = 6373.0
 
    ref_lat = radians(ref_latitude)
    points_lat = radians(points_latitude)
    dlat = points_lat - ref_lat
 
    ref_lon = radians(ref_longitude)
    points_lon = radians(points_longitude)
    dlon = points_lon - ref_lon
 
    def haversine(theta):
        return sin(theta / 2) ** 2
 
    central_angle = 2 * arcsin(sqrt(haversine(dlat) + cos(ref_lat) * cos(points_lat) * haversine(dlon)))
 
    return radius * central_angle * 1000

def afficher_carte(df_maps,df_defaut_final):
    m = folium.Map(location=[df_maps["latitude"].mean(), df_maps["longitude"].mean()], zoom_start=13)
    markers = folium.FeatureGroup(name="Markers")
    for _, row in df_maps.iterrows():
    # Ajouter un point à la localisation
        markers.add_child(
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,  # Taille du point
            color="black",  # Couleur du contour
            fill=True,
            fill_color="black",  # Couleur du point
            fill_opacity=0.7
        )
    )
    for _, row in df_maps.iterrows():
        markers.add_child(
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                icon=folium.DivIcon(
                    icon_size=(150, 36),
                    icon_anchor=(7, 7),
                    html=f'<div style="font-size: 14px; color: blue; font-weight: bold;">{int(row["Regard amont"])}</div>'
                )
            )
        )
    for _, row in df_defaut_final.iterrows():
        marker_icon = get_marker_icon(row["Defaut"],row["Gravite"])
        
        # Ajouter un marqueur avec l'icône personnalisée
        markers.add_child(
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                icon=marker_icon
            )
        )
    # Ajouter la FeatureGroup à la carte
    m.add_child(markers)

    # Ajouter une polyligne reliant les points
    coordinates = list(zip(df_maps['latitude'], df_maps['longitude']))
    folium.PolyLine(
        locations=coordinates,
        color='blue',
        weight=3,
        opacity=0.7
    ).add_to(m)




    return m

def get_color(gravity):
    if gravity == 3:
        return "red"
    elif gravity == 2:
        return "orange"
    elif gravity == 1:
        return "green"
    else:
        return "black"


def get_marker_icon(defaut,gravite):
    couleur = get_color(gravite)
    if defaut.lower() == "racine":
        return folium.Icon(icon="leaf", prefix="fa", color=couleur)  # Icône d'arbre (feuille)
    elif defaut.lower() == "intrusion":
        return folium.Icon(icon="tint", prefix="fa", color=couleur)  # Icône d'alerte
    elif defaut.lower() == "fissure":
        return folium.Icon(icon="bolt", prefix="fa", color=couleur)  # Icône de fissure (ou une autre icône)
    elif defaut.lower() == "infiltration":
        return folium.Icon(icon="water", prefix="fa", color=couleur)  # Icône de fissure (ou une autre icône)
    elif defaut.lower() == "depot":
        return folium.Icon(icon="map-marker-alt", prefix="fa", color=couleur)  # Icône de fissure (ou une autre icône)
    elif defaut.lower() == "concretion":
        return folium.Icon(icon="lock", prefix="fa", color=couleur)  # Icône de fissure (ou une autre icône)
    else:
        return None

# Titre de l'application
st.title("Traitement Sewerball Camera - Localisation des défauts")


uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])
st.write("""
### Prérequis pour le fichier CSV :
1. **Format CSV Standard** : Les valeurs doivent être séparées par point virgule (`;`).
2. **En-têtes de Colonnes** : La première ligne doit contenir les noms des colonnes, les colonnes obligatoires:["temps_video","Latitude","Longitude"].
3. **La première ligne** : Assurez-vous que la première ligne contienne l'heure de départ de la sewerball camera.
4. **Encodage** : Utilisez un encodage UTF-8 pour éviter les problèmes avec des caractères spéciaux.
         """)
if uploaded_file is not None:

    try:
        input_file = pd.read_csv(uploaded_file, sep=";", encoding="ISO-8859-1")
        systeme = st.selectbox(
        "Choisissez le système de coordonnées d'entrée",
        ("WGS84 (Latitude, Longitude)", "Lambert 93 (X, Y)")
        )
        if systeme == "Lambert 93 (X, Y)":
            input_file["Longitude"],input_file["Latitude"]=lambert93_to_wgs84(input_file["X"],input_file["Y"])

        copy_input=input_file.copy()
        df1 = copy_input.dropna(subset=['Latitude', 'Longitude'])
        df1=df1.reset_index(drop=True)
        
        # Séparer les lignes contenant defaut
        df2 = input_file.dropna(subset=['Defaut'])
        df2=df2.reset_index(drop=True)
        df2["temps_video"] = pd.to_timedelta(df2["temps_video"])
        df2 = df2[["temps_video", "Defaut", "Gravite"]]
        st.subheader("Données initiales")
        st.write(df1)
        st.subheader("Données des défauts")
        st.write(df2)

        try:
            df_selection=df1[["temps_video","Latitude","Longitude"]]
        except Exception as e:
            st.error(f"Les colonnes nécessaires ne sont disponibles ou pas au bon format.")
        st.success("Fichier chargé avec succès !")
    except Exception as e:
        st.error(f"Erreur inattendue : {e}")

            
    df_selection = df_selection.fillna(0)
    
    distances=[]
    azimuts=[]
    for i in range(len(df_selection) - 1):
        lat1, lon1 = df_selection.iloc[i]['Latitude'], df_selection.iloc[i]['Longitude']
        lat2, lon2 = df_selection.iloc[i + 1]['Latitude'], df_selection.iloc[i + 1]['Longitude']
        distance=approx_flying_distance_in_m(lat1, lon1, lat2, lon2)
        azimut=calculate_bearing(lat1, lon1, lat2, lon2)
        distances.append(distance)
        azimuts.append(azimut)
        
        df_selection['Appro_dist'] = pd.Series([None] + distances)
        df_selection["bearing"]=pd.Series([None] + azimuts)


    df_selection = df_selection.fillna(0)
    df_selection["Distance"]=df_selection["Appro_dist"].cumsum()
    df_selection["temps_video"] = pd.to_timedelta(df_selection["temps_video"])
    df_selection.set_index("temps_video", inplace=True)
    df_resampled = df_selection.resample("1s").asfreq()
    df_resampled["Distance"] = df_resampled["Distance"].resample("s").interpolate()
    df_resampled["bearing"] = df_resampled["bearing"].fillna(method='bfill')
    df_resampled["latitude_shift"] = df_resampled["Latitude"].shift(1)
    df_resampled["longitude_shift"] = df_resampled["Longitude"].shift(1)
    
    for i in range(1,len(df_resampled)-1):
        if pd.isna(df_resampled["Latitude"][i]):
            df_resampled["latitude_shift"][i+1], df_resampled["longitude_shift"][i+1] = get_destination_lat_long(df_resampled["latitude_shift"][i], df_resampled["longitude_shift"][i],df_resampled["bearing"][i],(df_resampled["Distance"][i+1]-df_resampled["Distance"][i]))
        else:
            continue

    df_resampled["Latitude"] = df_resampled["latitude_shift"].shift(-1)
    df_resampled["Longitude"] = df_resampled["longitude_shift"].shift(-1)
    df_resampled.drop(columns=['Appro_dist','latitude_shift', 'longitude_shift','bearing'], inplace=True)
    
    st.subheader("Données après resampling et interpolation")
    st.dataframe(df_resampled)

    result_csv = df_resampled.to_csv(index=True,sep=";")
    st.download_button(
        label="Télécharger le fichier résultant",
        data=result_csv,
        file_name="data_manhole_resultant.csv",
        mime="text/csv"
    )


    df_maps = df1[["Regard amont", "Latitude", "Longitude"]]
    df_maps = df_maps.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
    df_defaut_final = pd.merge(df_resampled, df2, on='temps_video', how='left')

    df_defaut_final = df_defaut_final.dropna(subset=['Defaut'])
    df_defaut_final = df_defaut_final.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
    df_defaut_final = df_defaut_final[df_defaut_final['Defaut'].notna() & df_defaut_final['Gravite'].notna()]
    st.dataframe(df_defaut_final)
    m_result = afficher_carte(df_maps,df_defaut_final)
    st_folium(m_result, width=700, height=500)
