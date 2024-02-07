from flask import Flask, render_template, request, send_file
import model  # Import your ML model module

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map', methods=['POST'])
def display_map():
    query = request.form['query']  # Get user input from the form
    # Pass the user input to your ML model and get the result
    result=model.process_query(query)
    result.save('map.html')
    # Process the result as needed
    # Save the result to a file (e.g., map.html)
    '''
    with open('map.html', 'w') as f:
        f.write(result)
    # Send the map.html file as a response
    '''
    return send_file('map.html')

if __name__ == '__main__':
    app.run(debug=True)
