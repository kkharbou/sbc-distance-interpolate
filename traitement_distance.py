import pandas as pd
import streamlit as st
import logging
import math
from math import cos, sin, atan2, radians, degrees, asin
from numpy import sin, cos, arcsin, radians, sqrt
import numpy as np
import folium
from folium.plugins import MarkerCluster
from io import BytesIO

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


# Titre de l'application
st.title("Traitement SBC distance")

uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])
if uploaded_file is not None:
    try:
        input_file = pd.read_csv(uploaded_file, sep=";", encoding="ISO-8859-1")
        st.subheader("Données initiales")
        st.write(input_file)
        st.success("Fichier chargé avec succès !")
    except UnicodeDecodeError as e:
        st.error("Erreur de décodage du fichier. Essayez d'utiliser un autre encodage.")
    except pd.errors.ParserError as e:
        st.error("Erreur lors de la lecture des données. Vérifiez le format du fichier CSV.")
    except Exception as e:
        st.error(f"Erreur inattendue : {e}")

    try:
        df_selection=input_file[["temps_video","Latitude","Longitude"]]
    except Exception as e:
        if "temps_video" not in input_file.columns or "Latitude" not in input_file.columns or "Longitude" not in input_file.columns:
            st.error(f"Les colonnes nécessaires sont manquantes ou pas au bon format")
            

    df_selection = df_selection.fillna(0)
    st.subheader("Inputs")
    st.write(df_selection)
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

    # Convertir le DataFrame résultant en fichier CSV
    result_csv = df_resampled.to_csv(index=True,sep=";")
    st.download_button(
        label="Télécharger le fichier résultant",
        data=result_csv,
        file_name="data_manhole_resultant.csv",
        mime="text/csv"
    )
