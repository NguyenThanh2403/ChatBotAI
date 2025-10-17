from sqlalchemy import Column, String, Integer, DateTime, Text, JSON
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class File(Base):
    __tablename__ = 'files'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    file_type = Column(String)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    chunks = relationship('Chunk', back_populates='file')

class Chunk(Base):
    __tablename__ = 'chunks'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    file_id = Column(String, ForeignKey('files.id'))
    text = Column(Text)
    page = Column(Integer, nullable=True)
    offset = Column(Integer, nullable=True)
    meta = Column('metadata', JSON, default=dict)
    file = relationship('File', back_populates='chunks')