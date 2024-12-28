

from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Flask uygulaması başlat
app = Flask(__name__)

# Veriyi ve modeli yükleyelim
dff = pd.read_csv('/Users/ates/Desktop/wine+quality/winequalitywhite.csv')
X = dff.drop(['quality','citric acid','free sulfur dioxide'], axis=1)
y = dff['quality']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Ana sayfa
@app.route('/')
def index():
    return render_template('index.html')

# Tahmin sayfası
@app.route('/predict', methods=['POST'])
def predict():
    # Kullanıcıdan gelen verileri al
    features = [float(request.form['sabit_asitlik']),
                float(request.form['ucucu_asitlik']),
                float(request.form['artik_seker']),
                float(request.form['klorur']),
                float(request.form['toplam_sulfurdioxit']),
                float(request.form['yoğunluk']),
                float(request.form['ph']),
                float(request.form['sülfatlar']),
                float(request.form['alkol'])]
    
    # Model ile tahmin yap
    input_data = pd.DataFrame([features], columns=['fixed acidity', 'volatile acidity', 'residual sugar', 
                                                  'chlorides',  'total sulfur dioxide', 'density', 
                                                  'pH', 'sulphates', 'alcohol'])
    predicted_quality = model.predict(input_data)
    
    if predicted_quality[0] < 5:
        quality = "Kötü"
    else:
        quality = "İyi"
        
    return render_template('result.html', 
                           sabit_asitlik=features[0],
                           ucucu_asitlik=features[1],
                           artik_seker=features[2],
                           klorur=features[3],
                           toplam_sulfurdioxit=features[4],
                           yogunluk=features[5],
                           ph=features[6],
                           sülfatlar=features[7],
                           alkol=features[8],
                           prediction=quality)

if __name__ == '__main__':
    app.run(debug=True)

 