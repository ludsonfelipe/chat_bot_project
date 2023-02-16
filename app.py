from flask import Flask, request
from deployment.asktochat import AskToChat

app = Flask(__name__)

ask = AskToChat()

@app.route("/predict", methods=["POST"])
def predict():
    input_str = request.data.decode()
    chatbot = AskToChat()
    response = chatbot.chat_answer(input_str)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)

