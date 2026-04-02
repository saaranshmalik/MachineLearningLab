"""
Default entrypoint for the stock prediction project.

Runs the Flask frontend so the project has a single obvious command.
"""

from app import app


if __name__ == "__main__":
    app.run(debug=True)
