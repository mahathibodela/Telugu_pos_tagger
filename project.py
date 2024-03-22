
from flask import Flask, render_template, request, url_for
import numpy as np  
import pickle
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
tags2id={ 1:'CC',
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
 19: 'WQ'}

app = Flask(__name__)
model=pickle.load(open('models/model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
      text=request.form['text']
      new=text.split()
      print(new)
      print(text)
      decoded_input = text.encode('utf-8').decode('utf-8')
      print(decoded_input)
      print(type(text))
      for i in text:
        print(i)
      onehot_rep_input=[one_hot(text,2267)]
      input_codded=pad_sequences(onehot_rep_input,padding="post",maxlen=30)
      print(input_codded)
      print(len(onehot_rep_input[0]))
      m=len(onehot_rep_input[0])
      # print(onehot_rep_input)
      
      list_input=[]
      list_input=model.predict(input_codded)
      p=list_input
      p = np.argmax(p, axis=-1)
      ans=[]
      dict={}
      k=len(text)
      print(p)
      for i in range(0,m):
          ans.append(tags2id[p[0][i]])
          print(tags2id[p[0][i]])
          dict[new[i]]=tags2id[p[0][i]]
      # print(ans)
      
      return render_template('output.html',text=dict)
    else:
         return render_template('index.html')


if __name__ == '__main__':
   app.config['JSON_AS_ASCII'] = False
   app.run()