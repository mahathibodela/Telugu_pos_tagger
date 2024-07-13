
import numpy as np
import pickle
import tensorflow
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
import streamlit as st
columns = st.columns(2)
with columns[0]:
    with st.container(border=True):
        st.session_state.input_text = st.text_area('Enter your text:')
with columns[1]:
    with st.container(border=True):
        if st.button(label='Generate Tags'):
            with st.status('loading model', expanded=True):
                model=pickle.load(open('models/model.pkl','rb'))
                "Model loaded"
            with st.status('predicting POS tags', expanded = True):
                split_text = st.session_state.input_text.split()
                decoded_input = st.session_state.input_text.encode('utf-8').decode('utf-8')
                onehot_rep_input=[one_hot(st.session_state.input_text,2267)]
                m=len(onehot_rep_input[0])
                input_codded=pad_sequences(onehot_rep_input,padding="post",maxlen=30)
                list_input=model.predict(input_codded)
                list_input = np.argmax(list_input, axis=-1)
                hashmap = dict()
                for i in range(0,m):
                    hashmap[split_text[i]]=tags2id[list_input[0][i]]
                output_list = hashmap.items()
                output_list = [f'{item}' for item in output_list]
                output = ','.join(output_list)
                'The output being: '
                with st.container(border=True):
                    f'{output}'



