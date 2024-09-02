from src.app import db
from sqlalchemy import Column, Integer, String, Float, DateTime

class TrafficSign(db.Model):
    __tablename__ = 'traffic_signs'

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    width = Column(Integer)
    height = Column(Integer)
    class_id = Column(Integer)
    class_name = Column(String)

class ModelResult(db.Model):
    __tablename__ = 'model_results'

    id = Column(Integer, primary_key=True)
    model_name = Column(String, nullable=False)
    accuracy = Column(Float)
    f1_score = Column(Float)
    val_accuracy = Column(Float)
    val_f1_score = Column(Float)
    test_accuracy = Column(Float)
    test_f1_score = Column(Float)
    training_time = Column(Float)
    timestamp = Column(DateTime)