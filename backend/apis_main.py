from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import uuid4, UUID
from datetime import datetime, date, timedelta
import sqlite3
import hashlib
import secrets
from fastapi.middleware.cors import CORSMiddleware
import jwt
from contextlib import contextmanager
from fancy_logger.console_logger import get_fancy_logger

logger = get_fancy_logger()

app = FastAPI()

# Configuration
SECRET_KEY = secrets.token_urlsafe(32)  # Generate a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Security
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_NAME = "../.db/todo_app.db"

def init_db():
    """Initialize the database with required tables"""
    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create todos table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS todos (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                criticality TEXT NOT NULL,
                urgency TEXT NOT NULL,
                due_date DATE NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        conn.commit()

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row  # This allows us to access columns by name
    try:
        yield conn
    finally:
        conn.close()

# Initialize database on startup
init_db()

# Status options
STATUS_OPTIONS = {"open", "in-progress", "done"}

# Utility functions
def hash_password(password: str) -> str:
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return hash_password(password) == hashed_password

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> str:
    """Verify and decode JWT token, return user_id"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    token = credentials.credentials
    user_id = verify_token(token)
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return {"id": user["id"], "username": user["username"]}

# Pydantic models
class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    created_at: datetime

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    username: str

class TodoCreate(BaseModel):
    title: str
    description: Optional[str] = None
    criticality: str
    urgency: str
    due_date: date

class TodoUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    criticality: Optional[str] = None
    urgency: Optional[str] = None
    due_date: Optional[date] = None
    status: Optional[str] = Field(default=None)

class TodoItem(BaseModel):
    id: str
    title: str
    description: Optional[str]
    criticality: str
    urgency: str
    due_date: date
    status: str
    user_id: str
    created_at: datetime

# Authentication Routes

@app.post("/register", response_model=dict)
def register_user(user: UserCreate):
    """Register a new user"""
    user_id = str(uuid4())
    hashed_password = hash_password(user.password)
    
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (id, username, password_hash) VALUES (?, ?, ?)",
                (user_id, user.username, hashed_password)
            )
            conn.commit()
            return {"message": "User registered successfully", "user_id": user_id}
        except sqlite3.IntegrityError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )

@app.post("/login", response_model=LoginResponse)
def login_user(user: UserLogin):
    """Login user and return access token"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (user.username,)
        )
        db_user = cursor.fetchone()
        
        if not db_user or not verify_password(user.password, db_user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": db_user["id"]}, expires_delta=access_token_expires
        )
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user_id=db_user["id"],
            username=db_user["username"]
        )

@app.post("/logout")
def logout_user(current_user: dict = Depends(get_current_user)):
    """Logout user (client should discard the token)"""
    return {"message": "Successfully logged out"}

# Protected Todo Routes

@app.post("/todos", response_model=str)
def create_todo(todo: TodoCreate, current_user: dict = Depends(get_current_user)):
    """Create a new todo item for the authenticated user"""
    todo_id = str(uuid4())
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO todos (id, title, description, criticality, urgency, due_date, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (todo_id, todo.title, todo.description, todo.criticality, 
              todo.urgency, todo.due_date, current_user["id"]))
        conn.commit()
    
    return todo_id

@app.get("/todos", response_model=List[TodoItem])
def get_all_todos(current_user: dict = Depends(get_current_user)):
    """Get all todos for the authenticated user"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, description, criticality, urgency, due_date, status, user_id, created_at
            FROM todos WHERE user_id = ?
            ORDER BY created_at DESC
        """, (current_user["id"],))
        
        todos = []
        for row in cursor.fetchall():
            todos.append(TodoItem(
                id=row["id"],
                title=row["title"],
                description=row["description"],
                criticality=row["criticality"],
                urgency=row["urgency"],
                due_date=row["due_date"],
                status=row["status"],
                user_id=row["user_id"],
                created_at=row["created_at"]
            ))
        
        return todos

@app.get("/todos/{todo_id}", response_model=TodoItem)
def get_todo_by_id(todo_id: str, current_user: dict = Depends(get_current_user)):
    """Get a specific todo by ID for the authenticated user"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, description, criticality, urgency, due_date, status, user_id, created_at
            FROM todos WHERE id = ? AND user_id = ?
        """, (todo_id, current_user["id"]))
        
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Todo not found")
        
        return TodoItem(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            criticality=row["criticality"],
            urgency=row["urgency"],
            due_date=row["due_date"],
            status=row["status"],
            user_id=row["user_id"],
            created_at=row["created_at"]
        )

@app.patch("/todos/{todo_id}", response_model=TodoItem)
def update_todo(todo_id: str, updates: TodoUpdate, current_user: dict = Depends(get_current_user)):
    """Update a todo item for the authenticated user"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # First, check if the todo exists and belongs to the user
        cursor.execute(
            "SELECT id FROM todos WHERE id = ? AND user_id = ?",
            (todo_id, current_user["id"])
        )
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Todo not found")
        
        # Prepare update data
        update_data = updates.dict(exclude_unset=True)
        if "status" in update_data and update_data["status"] not in STATUS_OPTIONS:
            raise HTTPException(status_code=400, detail="Invalid status value")
        
        if update_data:
            # Build dynamic update query
            set_clause = ", ".join([f"{key} = ?" for key in update_data.keys()])
            values = list(update_data.values()) + [todo_id, current_user["id"]]
            
            cursor.execute(f"""
                UPDATE todos SET {set_clause}
                WHERE id = ? AND user_id = ?
            """, values)
            conn.commit()
        
        # Return updated todo
        return get_todo_by_id(todo_id, current_user)

@app.delete("/todos/{todo_id}")
def delete_todo(todo_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a todo item for the authenticated user"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # First, check if the todo exists and belongs to the user
        cursor.execute(
            "SELECT status FROM todos WHERE id = ? AND user_id = ?",
            (todo_id, current_user["id"])
        )
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Todo not found")
        
        # if row["status"] not in {"open", "in-progress"}:
        #     raise HTTPException(status_code=403, detail="Cannot delete completed item")
        
        # Delete the todo
        cursor.execute(
            "DELETE FROM todos WHERE id = ? AND user_id = ?",
            (todo_id, current_user["id"])
        )
        conn.commit()
        
        return {"message": "Todo deleted"}

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)