from flask import Flask, request
from markupsafe import escape

from backend import analyze_series

app = Flask(__name__)

# Basic HTML template for input and output
HTML_PAGE = """
<!doctype html>
<html>
  <head><title>Summation Convergence Solver</title></head>
  <body>
    <h1>AP Calc BC Summation Convergence Solver</h1>
    <p>Enter the general term a<sub>n</sub> of an infinite series Σ a<sub>n</sub> (from n=1 to ∞):</p>
    <form method="get">
      <input type="text" name="expr" size="40" placeholder="e.g. 1/(n*log(n)**2)"/>
      <button type="submit">Check Convergence</button>
    </form>
    %s
  </body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    expr = request.args.get("expr", "")  # get 'expr' parameter from URL query string
    if expr == "":
        # No expression entered yet: show the form only
        return HTML_PAGE % ""
    # If expression provided, analyze it
    result = analyze_series(expr)
    # Escape the input expression for safe display
    expr_display = escape(expr)
    # Display result
    output = (
        f"<h3>Series: ∑<sub>n=1</sub><sup>∞</sup> {expr_display}</h3>"
        f"<p><b>Result:</b> {escape(result)}</p>"
    )
    return HTML_PAGE % output

# Note: In a real deployment, you would run app.run() to start the server.
