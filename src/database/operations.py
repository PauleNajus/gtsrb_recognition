from sqlalchemy.orm import sessionmaker
from src.database.models import TrafficSign, ModelResult
from datetime import datetime

def create_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()

def add_traffic_sign(session, filename, width, height, class_id, class_name):
    sign = TrafficSign(filename=filename, width=width, height=height, 
                       class_id=class_id, class_name=class_name)
    session.add(sign)
    session.commit()

def add_model_result(session, model_name, accuracy, f1_score, val_accuracy, val_f1_score, test_accuracy, test_f1_score, training_time):
    result = ModelResult(
        model_name=model_name, 
        accuracy=accuracy, 
        f1_score=f1_score,
        val_accuracy=val_accuracy,
        val_f1_score=val_f1_score,
        test_accuracy=test_accuracy,
        test_f1_score=test_f1_score,
        training_time=training_time,
        timestamp=datetime.now()
    )
    session.add(result)
    session.commit()

def get_all_traffic_signs(session):
    return session.query(TrafficSign).all()

def get_all_model_results(session):
    return session.query(ModelResult).all()