#STEP 1. Importing Required Libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle



#STEP 2.Create flask app
flask_app = Flask(__name__) # initializes the Flask application.
#The __name__ argument helps Flask determine the location of the app.



#STEP 3.Loading the Pretrained Model
model = pickle.load(open("model.pkl", "rb"))
#"rb"==read binary(pickle saves file in binary format)



#STEP 4.Creating the Homepage Route
@flask_app.route("/")#Defines the homepage route.
def Home():
    return render_template("index.html") #function renders the index.html template.
#When a user visits the website, this function is executed, loading the index.html page.



#STEP 5.Creating the Prediction Route
#This route handles predictions.
@flask_app.route("/predict", methods = ["POST"])
#It only accepts POST requests (data is sent from the frontend).
def predict():
    float_features = [float(x) for x in request.form.values()] #Extracts form values (inputted by the user) from an HTML form.
    features = [np.array(float_features)] #Model expects a NumPy array as input.
    prediction = model.predict(features)
    #Uses the pre-trained model to predict the flower species.
    return render_template("index.html", prediction_text = "Whether the COVID-19 made any psychological impact {}".format(prediction))



#STEP 6.Running the Application
if __name__ == "__main__":#This ensures that the script runs only when executed directly, not when imported as a module in another script.
    flask_app.run(debug=True)  #Automatic reloading of the server on code changes
