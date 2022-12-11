from flask import Flask
from flask import request
from flask import jsonify

# pip install tflite_runtime
import tflite_runtime.interpreter as tflite
#
import joblib
#
from sklearn.preprocessing import StandardScaler

scaler = joblib.load('std_scaler.bin') 

interpreter= tflite.Interpreter(model_path='Best_Model_3.tflite')
interpreter.allocate_tensors()

input_index=interpreter.get_input_details()[0]['index']
output_index=interpreter.get_output_details()[0]['index']

classes=['DOKOL', 'SAFAVI', 'ROTANA', 'DEGLET', 'SOGAY', 'IRAQI', 'BERHI']
    

app=Flask('date_fruit_classification')

@app.route('/predict',methods=['POST'])
def predict():
    
    example = request.get_json()

    scaled_example= scaler.transform(example['features'])

    interpreter.set_tensor(input_index,scaled_example.astype('float32'))
    interpreter.invoke()
    preds=interpreter.get_tensor(output_index)
    
    float_predictions =preds[0].tolist()
    
    result = dict(zip(classes,float_predictions))

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=9696)