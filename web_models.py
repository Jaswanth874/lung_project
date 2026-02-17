import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, 'webapp.db')
DB_URL = f'sqlite:///{DB_PATH}'

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    email = Column(String(256), unique=True, nullable=False)
    role = Column(String(32), default='user')
    password_hash = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    scans = relationship('CTScan', back_populates='owner')


class CTScan(Base):
    __tablename__ = 'ct_scans'
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(512))
    file_path = Column(String(1024))
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    owner_id = Column(Integer, ForeignKey('users.id'))

    owner = relationship('User', back_populates='scans')
    detections = relationship('DetectionResult', back_populates='scan')
    reports = relationship('ClinicalReport', back_populates='scan')


class DetectionResult(Base):
    __tablename__ = 'detection_results'
    id = Column(Integer, primary_key=True, index=True)
    scan_id = Column(Integer, ForeignKey('ct_scans.id'))
    confidence_score = Column(Float)
    lesion_size = Column(Float, nullable=True)
    created_date = Column(DateTime, default=datetime.datetime.utcnow)
    boxes_text = Column(Text)

    scan = relationship('CTScan', back_populates='detections')


class ClinicalReport(Base):
    __tablename__ = 'clinical_reports'
    id = Column(Integer, primary_key=True, index=True)
    scan_id = Column(Integer, ForeignKey('ct_scans.id'))
    content = Column(Text)
    created_date = Column(DateTime, default=datetime.datetime.utcnow)

    scan = relationship('CTScan', back_populates='reports')


def init_db():
    os.makedirs(BASE_DIR, exist_ok=True)
    Base.metadata.create_all(bind=engine)


if __name__ == '__main__':
    init_db()
