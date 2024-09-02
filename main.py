from src.app import create_app, db
from flask_migrate import Migrate
from config import Config
from src.database.models import TrafficSign, ModelResult 
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = create_app(Config)
migrate = Migrate(app, db)

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

@app.cli.command("init_db")
def init_db():
    with app.app_context():
        db.create_all()
        print("Database initialized!")

if __name__ == '__main__':
    with app.app_context():
        db.drop_all()
        db.create_all() 
        print("Database tables dropped and recreated.")
    app.run(debug=True)