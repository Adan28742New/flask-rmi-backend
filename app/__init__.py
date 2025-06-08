# app/__init__.py
from flask import Flask
from app.routes import main

def create_app():
    app = Flask(__name__)
    app.register_blueprint(main)

    # Nuevo log: Confirma que la aplicación Flask se ha configurado
    print("INFO: app/__init__.py - Aplicación Flask configurada y blueprint registrado.")

    return app
