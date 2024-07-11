import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

tags2id = {
    1: 'CC',
    2: 'DEM',
    3: 'INTF',
    4: 'JJ',
    5: 'NN',
    6: 'NNP',
    7: 'NST',
    8: 'PRP',
    9: 'PSP',
    10: 'QC',
    11: 'QF',
    12: 'QO',
    13: 'RB',
    14: 'RDP',
    15: 'RP',
    16: 'SYM',
    17: 'UT',
    18: 'VM',
    19: 'WQ'
}

# Load your model
model = pickle.load(open('models/model.pkl', 'rb'))

def predict_tags(text):
    new = text.split()
    onehot_rep_input = [one_hot(text, 2267)]
    input_encoded = pad_sequences(onehot_rep_input, padding="post", maxlen=30)
    list_input = model.predict(input_encoded)
    p = np.argmax(list_input, axis=-1)[0]
    return {word: tags2id[tag_id] for word, tag_id in zip(new, p)}

def main():
    st.title('POS Tagging Prediction')

    text_input = st.text_area('Enter your text:')
    if st.button('Predict'):
        if text_input:
            predictions = predict_tags(text_input)
            st.write('Predicted Tags:')
            st.write(predictions)

if __name__ == '__main__':
    main()
