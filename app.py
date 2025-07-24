from flask import Flask, request, render_template, jsonify, send_from_directory
from model import predict_salary
import traceback

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

#-------------------------
@app.route('/templates/<filename>')
def serve_template_file(filename):
    return send_from_directory('templates', filename)
#--------------------------

@app.route('/predict', methods=['POST'])
def predict():
    try:


        data = {
                'experience_level': request.form.get('experience_level'),
                'employment_type': request.form.get('employment_type'),
                'job_title': request.form.get('job_title'),
                'remote_ratio': int(request.form.get('remote_ratio', 0)),
                'company_location': request.form.get('company_location'),
                'company_size': request.form.get('company_size')
        }
        # Get all expected fields

        
        result = predict_salary(data)
        return jsonify({
            'status': 'success',
            'prediction': f"${result['prediction']:,.2f}",
            'range': f"${result['range'][0]:,.2f} - ${result['range'][1]:,.2f}"
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)