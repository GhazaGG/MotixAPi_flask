from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
port = 3000

model = None

async def load_model():
    global model
    model = await tf.keras.models.load_model('./indobert_hoax_detection_model.h5')

async def validate(link):
    try:
        if not model:
            await load_model()

        prediction = await model.predict(np.array([link]))
        label = np.argmax(prediction)
        message = "Berita tidak terdeteksi sebagai hoax" if label == 0 else "Berita terdeteksi sebagai Hoax"

        return {
            "validate": label,
            "link": link,
            "message": message,
        }
    except tf.errors.NotFoundError as err:
        return {"error": str(err)}, 404
    except Exception as err:
        print(err)
        return {"error": str(err)}, 500

@app.route('/')
def index():
    return 'Server active! By Flask'

@app.route('/validate', methods=['POST'])
async def validate_endpoint():
    link = request.json.get('link')

    if not link:
        return jsonify({"error": "URL tidak boleh kosong"}), 400

    data = await validate(link)
    return jsonify({"data": data}), 200

if __name__ == '__main__':
    app.run(port=port, debug=True)
