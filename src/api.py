import pandas as pd
from flask import Flask, request
from asktochat import AskToChat
import pickle 

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    input_str = request.data.decode()
    chatbot = AskToChat()
    response = chatbot.chat_answer(input_str)
    return response

if __name__ == '__main__':
    app.run()

