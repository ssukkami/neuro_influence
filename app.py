from flask import Flask, render_template, request, jsonify
import webview
import threading
import sys
import time
from inference import Predictor

DEV_MODE = False

app = Flask(__name__)
predictor = None


def load_model():
    global predictor
    print("[SYSTEM] INITIALIZING NEURAL CORE...")
    try:
        predictor = Predictor()
        print("[SYSTEM] CORE STATUS: ACTIVE")
    except Exception as e:
        print(f"[CRITICAL ERROR] MODEL LOADING FAILED: {e}")
        sys.exit(1)


load_model()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        if predictor is None:
            return jsonify({"error": "NEURAL_CORE_NOT_INITIALIZED"}), 500

        data = request.get_json(force=True) or {}

        text         = (data.get('text') or '').strip()
        followers    = int(data.get('followers')  or 1000)
        following    = int(data.get('following')  or 500)
        num_posts    = int(data.get('num_posts')  or 100)
        account_type = (data.get('account_type') or 'CREATOR').upper()

        raw_avg = data.get('avg_likes')
        avg_likes = int(raw_avg) if raw_avg not in (None, '', 0, '0') else None

        if not text:
            return jsonify({"error": "INPUT_BUFFER_EMPTY"}), 400

        result = predictor.predict(
            text=text,
            followers=followers,
            following=following,
            num_posts=num_posts,
            account_type=account_type,
            avg_likes=avg_likes,
            return_explanations=True,
        )

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] ANALYSIS_PIPELINE_CRASH: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def start_server():
    app.run(port=5000, debug=False, threaded=True, use_reloader=False)


if __name__ == '__main__':
    if DEV_MODE:
        print("[MODE] DEVELOPER_ENVIRONMENT_ACTIVE")
        app.run(debug=True, port=5000)
    else:
        print("[MODE] DESKTOP_PRODUCTION_INITIATED")

        server_thread = threading.Thread(target=start_server)
        server_thread.daemon = True
        server_thread.start()

        print("[SYSTEM] WAITING FOR LOCALHOST...")
        time.sleep(2)

        webview.create_window(
            'NeuroInfluence AI',
            'http://127.0.0.1:5000',
            width=1280,
            height=800,
            background_color='#050505',
        )
        webview.start()