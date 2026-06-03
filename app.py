"""
Flask app untuk prediksi harga rumah.
Memuat house_model.pkl lalu menyediakan form input + hasil prediksi.
"""
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Muat model sekali saat startup
with open("house_model.pkl", "rb") as f:
    bundle = pickle.load(f)
MODEL = bundle["model"]
FEATURES = bundle["features"]

# Label & nilai contoh agar form mudah diisi (California Housing)
FEATURE_INFO = {
    "MedInc":     ("Median pendapatan blok (puluhan ribu USD)", 5.0),
    "HouseAge":   ("Umur rumah (tahun)", 25.0),
    "AveRooms":   ("Rata-rata jumlah ruangan", 6.0),
    "AveBedrms":  ("Rata-rata kamar tidur", 1.0),
    "Population": ("Populasi blok", 1200.0),
    "AveOccup":   ("Rata-rata penghuni per rumah", 3.0),
    "Latitude":   ("Latitude", 34.0),
    "Longitude":  ("Longitude", -118.0),
}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    values = {f: FEATURE_INFO.get(f, ("", 0.0))[1] for f in FEATURES}

    if request.method == "POST":
        try:
            row = []
            for f in FEATURES:
                v = float(request.form.get(f, 0))
                values[f] = v
                row.append(v)
            pred = MODEL.predict([row])[0]
            # target dalam ratusan ribu USD
            prediction = f"${pred * 100000:,.0f}"
        except ValueError:
            prediction = "Input tidak valid, masukkan angka."

    fields = [(f, FEATURE_INFO.get(f, (f, 0.0))[0], values[f]) for f in FEATURES]
    return render_template("index.html", fields=fields, prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
