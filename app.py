import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load your trained logistic regression model
with open('logistic.pkl', 'rb') as f:
    model = pickle.load(f)

# Modern, responsive HTML template with Tailwind CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Eligibility Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 text-slate-100 min-h-screen py-10 px-4 sm:px-6 lg:px-8 flex flex-col justify-between">

    <div class="max-w-4xl mx-auto w-full">
        <!-- Header -->
        <div class="text-center mb-10">
            <div class="inline-flex items-center justify-center p-3 bg-indigo-500/10 rounded-2xl mb-4 border border-indigo-500/20 shadow-inner">
                <svg class="w-8 h-8 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
            </div>
            <h1 class="text-4xl font-extrabold tracking-tight text-white sm:text-5xl">Loan Eligibility Portal</h1>
            <p class="mt-3 text-base text-slate-400 max-w-xl mx-auto">Fill in the applicant details below to instantly evaluate loan approval status using our machine learning model.</p>
        </div>

        <!-- Main Card -->
        <div class="bg-slate-900/80 backdrop-blur-xl border border-slate-800 rounded-3xl shadow-2xl overflow-hidden p-6 sm:p-10">
            <form action="/" method="POST" class="grid grid-cols-1 sm:grid-cols-2 gap-6">
                
                <!-- Gender -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Gender</label>
                    <select name="Gender" required class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition">
                        <option value="1">Male</option>
                        <option value="0">Female</option>
                    </select>
                </div>

                <!-- Married -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Marital Status</label>
                    <select name="Married" required class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition">
                        <option value="1">Yes (Married)</option>
                        <option value="0">No (Unmarried)</option>
                    </select>
                </div>

                <!-- Dependents -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Dependents</label>
                    <select name="Dependents" required class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition">
                        <option value="0">0</option>
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3+</option>
                    </select>
                </div>

                <!-- Education -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Education</label>
                    <select name="Education" required class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition">
                        <option value="1">Graduate</option>
                        <option value="0">Not Graduate</option>
                    </select>
                </div>

                <!-- Self Employed -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Self Employed</label>
                    <select name="Self_Employed" required class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition">
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>

                <!-- Credit History -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Credit History (Satisfactory)</label>
                    <select name="Credit_History" required class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition">
                        <option value="1">Yes (1.0)</option>
                        <option value="0">No (0.0)</option>
                    </select>
                </div>

                <!-- Applicant Income -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Applicant Income ($)</label>
                    <input type="number" step="any" name="ApplicantIncome" placeholder="e.g. 5000" required class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition">
                </div>

                <!-- Coapplicant Income -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Coapplicant Income ($)</label>
                    <input type="number" step="any" name="CoapplicantIncome" placeholder="e.g. 1500" required class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition">
                </div>

                <!-- Loan Amount -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Loan Amount (in thousands)</label>
                    <input type="number" step="any" name="LoanAmount" placeholder="e.g. 120" required class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition">
                </div>

                <!-- Loan Amount Term -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Loan Amount Term (Months)</label>
                    <input type="number" step="any" name="Loan_Amount_Term" placeholder="e.g. 360" value="360" required class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition">
                </div>

                <!-- Property Area -->
                <div class="sm:col-span-2">
                    <label class="block text-sm font-medium text-slate-300 mb-2">Property Area</label>
                    <select name="Property_Area" required class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition">
                        <option value="2">Urban</option>
                        <option value="1">Semiurban</option>
                        <option value="0">Rural</option>
                    </select>
                </div>

                <!-- Submit Button -->
                <div class="sm:col-span-2 mt-4">
                    <button type="submit" class="w-full bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-semibold py-4 rounded-xl shadow-lg hover:shadow-indigo-500/25 transition duration-200 text-lg cursor-pointer">
                        Check Eligibility
                    </button>
                </div>
            </form>

            <!-- Result Box -->
            {% if prediction is not none %}
            <div class="mt-8 p-6 rounded-2xl border text-center transition-all duration-300 
                {% if prediction == 1 %} 
                    bg-emerald-950/40 border-emerald-500/30 text-emerald-300 
                {% else %} 
                    bg-rose-950/40 border-rose-500/30 text-rose-300 
                {% endif %}">
                <h3 class="text-xs uppercase tracking-widest font-semibold opacity-75 mb-1">Prediction Result</h3>
                <p class="text-2xl font-bold">
                    {% if prediction == 1 %}
                        🎉 Congratulations! Your Loan is likely to be <span class="text-emerald-400 underline decoration-2">APPROVED</span>.
                    {% else %}
                        ⚠️ Sorry, your Loan application is likely to be <span class="text-rose-400 underline decoration-2">REJECTED</span>.
                    {% endif %}
                </p>
            </div>
            {% endif %}
        </div>
    </div>

    <!-- Footer -->
    <footer class="mt-12 text-center text-xs text-slate-500">
        Powered by Flask & Machine Learning • Deployed on Render
    </footer>

</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Features map precisely to model expectation:
            # ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
            #  'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
            #  'Loan_Amount_Term', 'Credit_History', 'Property_Area']
            features = [
                float(request.form['Gender']),
                float(request.form['Married']),
                float(request.form['Dependents']),
                float(request.form['Education']),
                float(request.form['Self_Employed']),
                float(request.form['ApplicantIncome']),
                float(request.form['CoapplicantIncome']),
                float(request.form['LoanAmount']),
                float(request.form['Loan_Amount_Term']),
                float(request.form['Credit_History']),
                float(request.form['Property_Area'])
            ]
            
            final_features = np.array([features])
            prediction = int(model.predict(final_features)[0])
        except Exception as e:
            print(f"Error during prediction: {e}")
            prediction = None
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
