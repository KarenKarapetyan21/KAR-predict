import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Մոդելի բեռնում
with open('gradient_boosting_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Տվյալների ներբեռնում
df = pd.read_csv('NEW_DIGITAL_data.csv')

# Համայնքների կոդավորված արժեքներ
community_mapping = {
    "Կենտրոն": 449256,
    "Արաբկիր": 305770,
    "Նոր Նորք": 242278,
    "Մալաթիա Սեբաստիա": 258213,
    "Շենգավիթ": 230312,
    "Քանաքեռ Զեյթուն": 326437,
    "Ավան": 272642,
    "Դավթաշեն": 306828,
    "Էրեբունի": 239577,
    "Աջափնյակ": 249600,
    "Նուբարաշեն": 335682,
    "Նորք-Մարաշ": 358430
}

# Հավելվածի վերնագիր
st.title('Բնակարանի Ամսեկան Վարձի Կանխատեսման Հավելված')

# Մուտքային տվյալներ
st.header('Մուտքագրեք բնակարանի տվյալները')
total_area = st.number_input("Ընդհանուր մակերես (ք.մ)", min_value=10, max_value=1000, value=50)
room_count = st.number_input("Սենյակների քանակ", min_value=1, max_value=10, value=2)
floor = st.number_input("Հարկ", min_value=1, max_value=50, value=1)
building_type = st.selectbox("Շինության տիպ", ["Քարե", "Պանելային", "Մոնոլիտ", "Աղյուսե", "Կասետային", "Փայտե"])
renovation = st.selectbox("Վերանորոգում", ["Կապիտալ վերանորոգված", "Դիզայներական ոճով վերանորոգված", "Եվրովերանորոգված", "Կոսմետիկ վերանորոգում", "Մասնակի վերանորոգում", "Հին վերանորոգում", "Չվերանորոգված"])
new_building = st.selectbox("Նորակառույց", ["Այո", "Ոչ"])
furniture = st.selectbox("Կահույք", ["Առկա է", "Առկա չէ"])
community = st.selectbox("Համայնք", list(community_mapping.keys()))

# Համայնքի կոդավորված արժեքի ստացում
if community in community_mapping:
    community_code = community_mapping[community]
    
    # Ընտրում ենք տվյալները ըստ համայնքի կոդավորված արժեքի
    community_data = df[df['Համայնք'] == community_code]
    
    # Ստուգում ենք, արդյոք տվյալներ կան համայնքի համար
    if not community_data.empty:
        price_per_sq_meter = community_data['Գին_մեկ_մետրի_համար'].mean()
    else:
        st.warning(f"Ընտրված համայնքի `{community}` համար տվյալներ չկան: Հաշվարկն անկարելի է։")
        price_per_sq_meter = st.number_input("Գին մեկ քառակուսի մետրի համար", min_value=500, max_value=10000, value=3000)
else:
    st.warning("Համայնքը սխալ է նշված կամ չկա մատյանում։")
    price_per_sq_meter = st.number_input("Գին մեկ քառակուսի մետրի համար", min_value=500, max_value=10000, value=3000)

# Մեկ սենյակին բաժին ընկնող մակերեսի հաշվարկ
if room_count > 0:
    area_per_room = total_area / room_count
else:
    st.error("Սենյակների քանակը պետք է լինի 1 կամ ավելի։")
    area_per_room = 0

# Ավտոմատ արժեք՝ հին հայտարարություն
old_announcement = 1

# Փոխակերպում թվային արժեքների
building_type_mapping = {"Քարե": 1, "Պանելային": 2, "Մոնոլիտ": 3, "Աղյուսե": 4, "Կասետային": 5, "Փայտե": 6}
renovation_mapping = {"Կապիտալ վերանորոգված": 1, "Դիզայներական ոճով վերանորոգված": 2, "Եվրովերանորոգված": 3, "Կոսմետիկ վերանորոգում": 4, "Մասնակի վերանորոգում": 5, "Հին վերանորոգում": 6, "Չվերանորոգված": 7}
new_building_mapping = {"Այո": 1, "Ոչ": 0}
furniture_mapping = {"Առկա է": 1, "Առկա չէ": 0}

building_type_encoded = building_type_mapping[building_type]
renovation_encoded = renovation_mapping[renovation]
new_building_encoded = new_building_mapping[new_building]
furniture_encoded = furniture_mapping[furniture]

# Կանխատեսում
if st.button("Կանխատեսել վարձը"):
    features = np.array([[total_area, room_count, floor, building_type_encoded, renovation_encoded,
                          new_building_encoded, furniture, community_code, price_per_sq_meter, area_per_room, old_announcement]])
    prediction = model.predict(features)[0]

    # Կլորացնում ենք 10,000-ի հաշվարկով
    rounded_prediction = round(prediction / 10000) * 10000

    st.subheader(f"Բնակարանի կանխատեսված ամսեկան վարձը՝ {rounded_prediction:,.0f} դրամ։")
