"""
Database module for managing MySQL interactions for the faculty information chatbot.
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database connection parameters for XAMPP
DB_HOST = "localhost" 
DB_USER = "root"
DB_PASSWORD = ""  # Default XAMPP has no password for root
DB_NAME = "faculty_chatbot"
DB_PORT = 3306

# Create SQLAlchemy engine with XAMPP MySQL connection string
engine = create_engine(f"mysql+pymysql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for database models
Base = declarative_base()

# Define database models
class QuestionEvaluation(Base):
    """Model for storing question evaluations."""
    
    __tablename__ = "questions_evaluation"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    liked = Column(Boolean, nullable=True)  # True for like, False for dislike, None if not rated
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<QuestionEvaluation(id={self.id}, question='{self.question[:20]}...', liked={self.liked})>"


def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(bind=engine)


def store_question_answer(session_id, question, answer):
    """
    Store a question and its answer in the database.
    
    Args:
        session_id: The unique session ID
        question: The user's question
        answer: The chatbot's answer
        
    Returns:
        The ID of the created record
    """
    db = SessionLocal()
    try:
        eval_entry = QuestionEvaluation(
            session_id=session_id,
            question=question,
            answer=answer,
            liked=None  # Initially set to None (not rated)
        )
        db.add(eval_entry)
        db.commit()
        db.refresh(eval_entry)
        return eval_entry.id
    finally:
        db.close()


def update_evaluation(evaluation_id, liked):
    """
    Update the evaluation (like/dislike) for a question.
    
    Args:
        evaluation_id: The ID of the evaluation record
        liked: Boolean indicating whether the answer was liked (True) or disliked (False)
        
    Returns:
        True if successful, False otherwise
    """
    db = SessionLocal()
    try:
        eval_entry = db.query(QuestionEvaluation).filter(QuestionEvaluation.id == evaluation_id).first()
        if eval_entry:
            eval_entry.liked = liked
            db.commit()
            return True
        return False
    finally:
        db.close()


def get_evaluation_stats():
    """
    Get statistics about evaluations.
    
    Returns:
        Dictionary containing evaluation statistics
    """
    db = SessionLocal()
    try:
        total = db.query(QuestionEvaluation).count()
        liked = db.query(QuestionEvaluation).filter(QuestionEvaluation.liked == True).count()
        disliked = db.query(QuestionEvaluation).filter(QuestionEvaluation.liked == False).count()
        not_rated = db.query(QuestionEvaluation).filter(QuestionEvaluation.liked == None).count()
        
        return {
            "total": total,
            "liked": liked,
            "disliked": disliked,
            "not_rated": not_rated,
            "liked_percentage": round(liked / total * 100, 2) if total > 0 else 0
        }
    finally:
        db.close()