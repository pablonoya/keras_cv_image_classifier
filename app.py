import streamlit as st
import tensorflow as tf
from zipfile import ZipFile
import matplotlib.pyplot as plt


@st.cache_data
def get_class_names():
    return [
        "chihuahua",
        "japanese_spaniel",
        "maltese_dog",
        "pekinese",
        "shih",
        "blenheim_spaniel",
        "papillon",
        "toy_terrier",
        "rhodesian_ridgeback",
        "afghan_hound",
        "basset",
        "beagle",
        "bloodhound",
        "bluetick",
        "black",
        "walker_hound",
        "english_foxhound",
        "redbone",
        "borzoi",
        "irish_wolfhound",
        "italian_greyhound",
        "whippet",
        "ibizan_hound",
        "norwegian_elkhound",
        "otterhound",
        "saluki",
        "scottish_deerhound",
        "weimaraner",
        "staffordshire_bullterrier",
        "american_staffordshire_terrier",
        "bedlington_terrier",
        "border_terrier",
        "kerry_blue_terrier",
        "irish_terrier",
        "norfolk_terrier",
        "norwich_terrier",
        "yorkshire_terrier",
        "wire",
        "lakeland_terrier",
        "sealyham_terrier",
        "airedale",
        "cairn",
        "australian_terrier",
        "dandie_dinmont",
        "boston_bull",
        "miniature_schnauzer",
        "giant_schnauzer",
        "standard_schnauzer",
        "scotch_terrier",
        "tibetan_terrier",
        "silky_terrier",
        "soft",
        "west_highland_white_terrier",
        "lhasa",
        "flat",
        "curly",
        "golden_retriever",
        "labrador_retriever",
        "chesapeake_bay_retriever",
        "german_short",
        "vizsla",
        "english_setter",
        "irish_setter",
        "gordon_setter",
        "brittany_spaniel",
        "clumber",
        "english_springer",
        "welsh_springer_spaniel",
        "cocker_spaniel",
        "sussex_spaniel",
        "irish_water_spaniel",
        "kuvasz",
        "schipperke",
        "groenendael",
        "malinois",
        "briard",
        "kelpie",
        "komondor",
        "old_english_sheepdog",
        "shetland_sheepdog",
        "collie",
        "border_collie",
        "bouvier_des_flandres",
        "rottweiler",
        "german_shepherd",
        "doberman",
        "miniature_pinscher",
        "greater_swiss_mountain_dog",
        "bernese_mountain_dog",
        "appenzeller",
        "entlebucher",
        "boxer",
        "bull_mastiff",
        "tibetan_mastiff",
        "french_bulldog",
        "great_dane",
        "saint_bernard",
        "eskimo_dog",
        "malamute",
        "siberian_husky",
        "affenpinscher",
        "basenji",
        "pug",
        "leonberg",
        "newfoundland",
        "great_pyrenees",
        "samoyed",
        "pomeranian",
        "chow",
        "keeshond",
        "brabancon_griffon",
        "pembroke",
        "cardigan",
        "toy_poodle",
        "miniature_poodle",
        "standard_poodle",
        "mexican_hairless",
        "dingo",
        "dhole",
        "african_hunting_dog",
    ]


@st.cache_resource(show_spinner="Cargando modelo...")
def load_model(model_file):
    with ZipFile(model_file) as zipf:
        zipf.extractall("model")

    return tf.saved_model.load("model")


def plot_results(probs, classes):
    fig, ax = plt.subplots()

    ax.barh(classes, probs, color=["grey", "grey", "royalblue"])
    ax.set_xlabel("Probabilidad")
    ax.set_xticks([0, 25, 50, 75, 100])

    st.pyplot(fig)


def main():
    st.title("Clasificación de razas de perros")
    st.markdown(
        "Repositorio en  [GitHub](https://github.com/pablonoya/keras_cv_image_classifier)"
    )
    st.subheader("Carga un modelo exportado")

    model_file = st.file_uploader("Sube un zip del modelo exportado", type="zip")
    if not model_file:
        return

    model = load_model(model_file)

    st.subheader("Carga una imagen")
    image_file = st.file_uploader("Sube la imagen de un perrito")
    if not image_file:
        return

    st.subheader("Resultados 🐶")

    image = tf.keras.utils.load_img(image_file, target_size=(224, 224))
    image_arr = tf.keras.utils.img_to_array(image) / 255.0
    image_arr = tf.expand_dims(image_arr, axis=0)

    pred = model.serve(image_arr)

    # Get top 3 predictions sorted by probability
    values, indices = tf.math.top_k(pred, k=3, sorted=True)

    # Sort probabilities and classes in descending order
    probs = values.numpy()[0][::-1] * 100
    indices = indices.numpy()[0][::-1]

    classes = []
    for i, idx in enumerate(indices):
        class_name = get_class_names()[idx]
        prob = probs[i]
        classes.append(f"{class_name} ({prob :.1f}%)")

    # Plot results
    st.image(image_file, caption=classes[-1])
    plot_results(probs, classes)


if __name__ == "__main__":
    main()
