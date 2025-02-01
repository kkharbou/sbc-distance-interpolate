import pandas as pd
import streamlit as st

# Titre de l'application
st.title("Traitement SBC distance")

# 1. Charger le fichier CSV via Streamlit
uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])
if uploaded_file is not None:
    # Charger le fichier CSV en utilisant pandas
    input_file = pd.read_csv(uploaded_file, sep=";")
    st.subheader("Données initiales")
    st.write(input_file)


    df_selection=input_file[["temps video","LONGUEUR_1"]]
    df_selection = df_selection.fillna(0)
    st.subheader("Colonnes temps et distance")
    st.write(df_selection)


    df_selection["LONGUEUR_CUMULATIVE"]=df_selection["LONGUEUR_1"].cumsum()
    df_selection["temps video"] = pd.to_timedelta(df_selection["temps video"])
    df_selection.set_index("temps video", inplace=True)


    df_resampled = df_selection.resample("s").interpolate()
    df_resampled["LONGUEUR_CUMULATIVE"] = round(df_resampled["LONGUEUR_CUMULATIVE"],3)
    df_resampled = df_resampled["LONGUEUR_CUMULATIVE"]
    df_resampled['Date']= df_resampled.index


    st.subheader("Données après resampling et interpolation")
    st.dataframe(df_resampled)

    # Convertir le DataFrame résultant en fichier CSV
    result_csv = df_resampled.to_csv(index=True, sep=";")
    st.download_button(
        label="Télécharger le fichier résultant",
        data=result_csv,
        file_name="data_manhole_resultant.csv",
        mime="text/csv"
    )
    # df_resampled.to_csv(r"C:\Users\LEB953\workspace\swmm_tss\swmm_tss\data_manhole_resultant.csv")
    # # input_file['temps video'] = pd.to_datetime(input_file['temps video'])

    # # input_file['duree'] = input_file['temps video'].diff().dt.total_seconds()
    # # input_file["distance parcourue par seconde"]=input_file['LONGUEUR_1']/input_file['duree']

    # # print(input_file)
    # # # Supposons que les dates sont dans la première colonne et successives
    # # resultats = []

    # # for i in range(0, len(input_file["temps video"])-1, 1):  # Parcours des dates deux par deux
    # #     print(i)
    # #     date_debut = pd.to_datetime(input_file.iloc[i, 13])
    # #     date_fin = pd.to_datetime(input_file.iloc[i + 1, 13])
    # #     print(date_debut, date_fin)
    # #     df_temp = pd.DataFrame({
    # #         "timestamp": pd.date_range(start=date_debut, end=date_fin, freq="s"),
    # #         "valeur": input_file.iloc[i+1, 15] + input_file.iloc[i+1, 15] * pd.Series(range(len(pd.date_range(start=date_debut, end=date_fin, freq="S"))))
    # #     })
    # #     print(df_temp)
    # # #     resultats.append(df_temp)
    # # #     print(resultats)

    # # # print(resultats)