# FastAPI Complete Learning Guide - Build a Task Management API

## 📌 Project Overview

This guide will teach you **all FastAPI concepts** through a practical **Task Management API** project. You’ll build a RESTful API with:

- ✅ CRUD operations (Create, Read, Update, Delete)
- ✅ Database integration (SQLite)
- ✅ Request/Response validation
- ✅ Error handling
- ✅ Authentication basics
- ✅ Middleware
- ✅ CORS support
- ✅ Type hints and documentation

-----

## 🔧 Prerequisites

```bash
# Install required packages
pip install fastapi
pip install uvicorn
pip install sqlalchemy
pip install pydantic
pip install python-dotenv
```

-----

## 📁 Project Structure

```
task-api/
├── main.py              # Main FastAPI application
├── models.py            # Database models
├── database.py          # Database configuration
├── schemas.py           # Pydantic models (validation)
├── crud.py              # Database operations
└── requirements.txt     # Dependencies
```

-----

## 🚀 Complete Code - Line by Line Explanation

### 1️⃣ **database.py** - Database Setup

```python
# ============================================================================
# DATABASE CONFIGURATION - Setup SQLAlchemy and create database session
# ============================================================================

# Import SQLAlchemy components for database operations
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# ============================================================================
# DATABASE URL
# ============================================================================
# This is the connection string for SQLite
# "sqlite:///./test.db" means:
#   - "sqlite:///" → Use SQLite database
#   - "./" → Current directory
#   - "test.db" → Database file name
DATABASE_URL = "sqlite:///./test.db"

# ============================================================================
# CREATE DATABASE ENGINE
# ============================================================================
# The engine is the core interface to the database
# create_engine() establishes connection parameters
# connect_args={"check_same_thread": False} → SQLite specific setting
#   - Allows multiple threads to access the database
#   - Only needed for SQLite (not for PostgreSQL/MySQL)
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# ============================================================================
# CREATE SESSION FACTORY
# ============================================================================
# SessionLocal is a factory that creates new database sessions
# Each request will get its own session to avoid conflicts
SessionLocal = sessionmaker(
    autocommit=False,      # Require explicit commit()
    autoflush=False,       # Don't auto-flush before queries
    bind=engine            # Bind this session to our engine
)

# ============================================================================
# CREATE BASE CLASS FOR MODELS
# ============================================================================
# Base is the parent class for all database models
# It provides the metadata container for all table definitions
Base = declarative_base()

# ============================================================================
# DEPENDENCY FUNCTION
# ============================================================================
# This function creates a database session for each request
# The "yield" keyword makes this a generator - session is created before the
# request and cleaned up after (finally block executes after yield)
def get_db():
    """
    Create a database session for a request
    
    Yields:
        Session: SQLAlchemy database session
    
    Example:
        This is used in FastAPI dependencies:
        def my_endpoint(db: Session = Depends(get_db)):
            # db is automatically injected here
    """
    db = SessionLocal()
    try:
        # Yield the session to the request handler
        yield db
    finally:
        # This always runs after the request completes
        # Closes the session to free up resources
        db.close()
```

-----

### 2️⃣ **models.py** - Database Models (SQLAlchemy)

```python
# ============================================================================
# DATABASE MODELS - Define table structure
# ============================================================================

# Import Column types and functions from SQLAlchemy
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from datetime import datetime

# Import Base class from database.py
from database import Base

# ============================================================================
# TASK MODEL
# ============================================================================
# This class represents the 'tasks' table in the database
# Each attribute becomes a column in the table
class Task(Base):
    """
    Task database model
    Maps to 'tasks' table in SQLite database
    
    Attributes:
        id: Unique identifier (Primary Key)
        title: Task title/name
        description: Detailed task description
        completed: Task completion status
        created_at: Timestamp when task was created
    """
    
    # This special attribute tells SQLAlchemy what table name to use
    __tablename__ = "tasks"
    
    # ========================================================================
    # Column Definitions
    # ========================================================================
    
    # Primary Key: Unique identifier, auto-increments
    # Integer: This column stores integer values
    # primary_key=True: This is the unique identifier for each row
    # index=True: Create an index for faster queries
    id = Column(
        Integer,
        primary_key=True,
        index=True
    )
    
    # String column for task title
    # String(100): Maximum 100 characters
    # nullable=False: This field is REQUIRED (cannot be null/empty)
    # index=True: Create index for faster searching by title
    title = Column(
        String(100),
        nullable=False,
        index=True
    )
    
    # String column for task description
    # String(500): Maximum 500 characters
    # nullable=True: This field is optional (can be empty)
    description = Column(
        String(500),
        nullable=True
    )
    
    # Boolean column for completion status
    # Boolean: True/False values
    # default=False: By default, tasks are not completed
    # nullable=False: This field is REQUIRED
    completed = Column(
        Boolean,
        default=False,
        nullable=False
    )
    
    # DateTime column for creation timestamp
    # DateTime: Stores date and time
    # func.now(): Use database's current timestamp function
    # server_default: Set the value at database level
    # This ensures consistent timestamps across timezones
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
```

-----

### 3️⃣ **schemas.py** - Pydantic Models (Validation & Serialization)

```python
# ============================================================================
# PYDANTIC SCHEMAS - Request/Response validation
# ============================================================================

# Pydantic is used for:
# 1. Input validation - Check request data is correct type/format
# 2. Serialization - Convert database models to JSON
# 3. Auto-documentation - Generate OpenAPI docs

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# ============================================================================
# CREATE TASK SCHEMA (Request Body)
# ============================================================================
# This schema validates data when creating a new task
# Only includes fields that client provides
class TaskCreate(BaseModel):
    """
    Schema for creating a new task
    
    Fields:
        title: Task title (required)
        description: Task description (optional)
    """
    
    # Field() allows adding validation rules and descriptions
    # min_length=1: Title must be at least 1 character
    # max_length=100: Title cannot exceed 100 characters
    title: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Task title"
    )
    
    # Optional field - client doesn't need to provide it
    # default=None: If not provided, it will be None
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Task description"
    )


# ============================================================================
# UPDATE TASK SCHEMA (Request Body)
# ============================================================================
# This schema validates data when updating a task
# All fields are optional (client only provides what they want to update)
class TaskUpdate(BaseModel):
    """
    Schema for updating an existing task
    
    All fields are optional - provide only what you want to update
    """
    
    # Optional field for updating title
    title: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Updated task title"
    )
    
    # Optional field for updating description
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Updated task description"
    )
    
    # Optional field for updating completion status
    completed: Optional[bool] = Field(
        default=None,
        description="Task completion status"
    )


# ============================================================================
# TASK RESPONSE SCHEMA
# ============================================================================
# This schema defines what data is sent back to the client
# Includes all fields from database + computed fields
class TaskResponse(BaseModel):
    """
    Schema for task responses from the API
    
    Used to serialize Task database model to JSON
    Includes all task information
    """
    
    # All fields from database model
    id: int
    title: str
    description: Optional[str]
    completed: bool
    created_at: datetime
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    class Config:
        """
        Pydantic configuration for this model
        
        from_attributes=True: Allow creating this schema from ORM objects
        This means Pydantic can read attributes from SQLAlchemy models
        """
        from_attributes = True
        
        # json_schema_extra provides example data for API documentation
        json_schema_extra = {
            "example": {
                "id": 1,
                "title": "Learn FastAPI",
                "description": "Complete FastAPI tutorial",
                "completed": False,
                "created_at": "2024-01-15T10:30:00"
            }
        }


# ============================================================================
# TASK LIST RESPONSE SCHEMA
# ============================================================================
# Used when returning a list of tasks with metadata
class TaskListResponse(BaseModel):
    """
    Schema for returning a list of tasks with pagination info
    """
    
    # List of task objects
    tasks: list[TaskResponse]
    
    # Total count of tasks
    total: int
    
    # Number of items in this response
    count: int
```

-----

### 4️⃣ **crud.py** - CRUD Operations (Database Queries)

```python
# ============================================================================
# CRUD OPERATIONS - Create, Read, Update, Delete functions
# ============================================================================

# These functions contain the business logic for database operations
# CRUD = Create, Read, Update, Delete

from sqlalchemy.orm import Session
from models import Task
from schemas import TaskCreate, TaskUpdate

# ============================================================================
# CREATE - Add new task to database
# ============================================================================
def create_task(db: Session, task: TaskCreate) -> Task:
    """
    Create a new task in the database
    
    Args:
        db (Session): Database session
        task (TaskCreate): Task data from request
    
    Returns:
        Task: Created task object with generated ID
    
    Explanation:
        1. Create a new Task object with provided data
        2. Add it to the database session
        3. Commit the transaction
        4. Refresh to get the auto-generated ID
        5. Return the created task
    """
    
    # Create a new Task database object
    # **task.model_dump() converts Pydantic model to dictionary
    db_task = Task(**task.model_dump())
    
    # Add this object to the session (staged for insert)
    db.add(db_task)
    
    # Commit the transaction (execute INSERT into database)
    db.commit()
    
    # Refresh the object to load generated values (like auto-increment ID)
    db.refresh(db_task)
    
    # Return the newly created task
    return db_task


# ============================================================================
# READ - Get tasks from database
# ============================================================================

def get_task(db: Session, task_id: int) -> Task:
    """
    Get a single task by ID
    
    Args:
        db (Session): Database session
        task_id (int): ID of task to retrieve
    
    Returns:
        Task: Task object or None if not found
    
    Explanation:
        1. Query tasks table
        2. Filter by ID (WHERE clause)
        3. Get first result
        4. Returns None if not found
    """
    
    # db.query(Task) - Create a query object
    # .filter(Task.id == task_id) - Add WHERE clause
    # .first() - Get first matching row (or None)
    return db.query(Task).filter(Task.id == task_id).first()


def get_all_tasks(
    db: Session,
    skip: int = 0,
    limit: int = 10,
    completed: bool = None
) -> list[Task]:
    """
    Get all tasks with optional filtering and pagination
    
    Args:
        db (Session): Database session
        skip (int): Number of records to skip (pagination offset)
        limit (int): Maximum records to return (pagination limit)
        completed (bool): Filter by completion status (optional)
    
    Returns:
        list[Task]: List of task objects
    
    Explanation:
        1. Start a query on Task table
        2. Add optional completion filter
        3. Apply pagination (skip and limit)
        4. Execute and return results
    """
    
    # Start building query
    query = db.query(Task)
    
    # If completed filter is provided, add it to query
    # This is optional filtering
    if completed is not None:
        query = query.filter(Task.completed == completed)
    
    # .offset() - Skip this many records
    # .limit() - Maximum records to return
    # .all() - Execute query and return all results
    return query.offset(skip).limit(limit).all()


def get_tasks_count(db: Session) -> int:
    """
    Get total number of tasks
    
    Args:
        db (Session): Database session
    
    Returns:
        int: Total count of tasks
    
    Explanation:
        1. Query the Task table
        2. Count all rows
        3. Return the count
    """
    
    # .count() - Returns number of rows matching the query
    return db.query(Task).count()


# ============================================================================
# UPDATE - Modify existing task
# ============================================================================

def update_task(
    db: Session,
    task_id: int,
    task_update: TaskUpdate
) -> Task:
    """
    Update an existing task
    
    Args:
        db (Session): Database session
        task_id (int): ID of task to update
        task_update (TaskUpdate): New data for the task
    
    Returns:
        Task: Updated task object or None if not found
    
    Explanation:
        1. Find the task by ID
        2. Update only provided fields
        3. Commit changes
        4. Return updated task
    """
    
    # Query and get the task
    db_task = db.query(Task).filter(Task.id == task_id).first()
    
    # If task not found, return None
    if not db_task:
        return None
    
    # Get the update data
    # model_dump(exclude_unset=True) returns only provided fields
    # exclude_unset=True means: only include fields that client provided
    update_data = task_update.model_dump(exclude_unset=True)
    
    # Update each field that was provided
    for key, value in update_data.items():
        setattr(db_task, key, value)
    
    # Commit the changes to database
    db.commit()
    
    # Refresh to get any database-level changes
    db.refresh(db_task)
    
    # Return the updated task
    return db_task


# ============================================================================
# DELETE - Remove task from database
# ============================================================================

def delete_task(db: Session, task_id: int) -> bool:
    """
    Delete a task by ID
    
    Args:
        db (Session): Database session
        task_id (int): ID of task to delete
    
    Returns:
        bool: True if deleted, False if not found
    
    Explanation:
        1. Find the task by ID
        2. Delete it from session
        3. Commit the deletion
        4. Return success status
    """
    
    # Query and get the task
    db_task = db.query(Task).filter(Task.id == task_id).first()
    
    # If task doesn't exist, return False
    if not db_task:
        return False
    
    # Remove from database session (stage for deletion)
    db.delete(db_task)
    
    # Commit the deletion
    db.commit()
    
    # Return True to indicate success
    return True
```

-----

### 5️⃣ **main.py** - FastAPI Application

```python
# ============================================================================
# MAIN FASTAPI APPLICATION
# ============================================================================

# This is the main entry point of your API
# Contains route definitions, middleware, and application configuration

# ============================================================================
# IMPORTS
# ============================================================================

from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Optional

# Import our custom modules
from database import engine, get_db, Base
from models import Task
from schemas import TaskCreate, TaskUpdate, TaskResponse, TaskListResponse
import crud

# ============================================================================
# CREATE DATABASE TABLES
# ============================================================================
# This creates all tables defined in models.py if they don't exist
# Base.metadata.create_all() runs CREATE TABLE IF NOT EXISTS for all models
Base.metadata.create_all(bind=engine)

# ============================================================================
# INITIALIZE FASTAPI APPLICATION
# ============================================================================
# FastAPI() creates the main application instance
# title: Name shown in API documentation
# description: Description in API docs
# version: API version
app = FastAPI(
    title="Task Management API",
    description="Complete FastAPI learning project - CRUD operations with database",
    version="1.0.0"
)

# ============================================================================
# MIDDLEWARE - CORS (Cross-Origin Resource Sharing)
# ============================================================================
# CORS allows your API to be accessed from different domains
# Without this, browsers block requests from different origins

# add_middleware() registers middleware to process all requests/responses
# CORSMiddleware: Handles CORS headers

app.add_middleware(
    CORSMiddleware,
    
    # allow_origins: Which origins (domains) can access this API
    # ["*"] means allow all origins (not recommended for production)
    # In production, specify exact domains like ["https://yourdomain.com"]
    allow_origins=["*"],
    
    # allow_credentials: Allow cookies and authorization headers
    allow_credentials=True,
    
    # allow_methods: Which HTTP methods are allowed
    # ["*"] means allow all (GET, POST, PUT, DELETE, etc.)
    allow_methods=["*"],
    
    # allow_headers: Which headers clients can send
    # ["*"] means allow all headers
    allow_headers=["*"],
)

# ============================================================================
# CUSTOM MIDDLEWARE - Request Logging
# ============================================================================
# Middleware runs before and after each request
# @app.middleware("http") registers HTTP middleware

@app.middleware("http")
async def add_process_time_header(request, call_next):
    """
    Middleware that adds response time to headers
    
    Args:
        request: The incoming HTTP request
        call_next: Function to call the next middleware/route
    
    Returns:
        response: Modified response with custom header
    
    Explanation:
        1. Get the starting time
        2. Call the route handler
        3. Calculate elapsed time
        4. Add time to response headers
        5. Return response
    """
    
    import time
    
    # Record start time
    start_time = time.time()
    
    # Call the route handler (returns response)
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Add custom header to response
    # This header will appear in response headers
    response.headers["X-Process-Time"] = str(process_time)
    
    # Return the modified response
    return response


# ============================================================================
# ROOT ENDPOINT - Health Check
# ============================================================================
# GET / - The simplest endpoint, returns a greeting

@app.get("/")
async def root():
    """
    Root endpoint - Health check
    
    Returns:
        dict: Simple greeting message
    
    Explanation:
        @app.get("/") - Decorator that registers this function as a GET route
        async def - Asynchronous function (handles concurrent requests)
        No parameters needed here
    """
    return {
        "message": "Welcome to Task Management API",
        "version": "1.0.0",
        "status": "running"
    }


# ============================================================================
# CREATE TASK ENDPOINT
# ============================================================================
# POST /tasks - Create a new task

@app.post(
    "/tasks",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new task",
    tags=["Tasks"]
)
def create_task_endpoint(
    task: TaskCreate,
    db: Session = Depends(get_db)
) -> TaskResponse:
    """
    Create a new task
    
    Args:
        task (TaskCreate): Task data from request body
        db (Session): Database session (injected by FastAPI)
    
    Returns:
        TaskResponse: The created task with ID
    
    Explanation:
        @app.post() - Registers this as a POST request handler
        response_model=TaskResponse - Validate response matches this schema
        status_code=201 - Return HTTP 201 Created status
        task: TaskCreate - FastAPI automatically validates request body
        db: Session = Depends(get_db) - Dependency injection of database session
    """
    
    # Call CRUD function to create task in database
    db_task = crud.create_task(db=db, task=task)
    
    # Return the created task
    # FastAPI automatically converts Task model to TaskResponse (JSON)
    return db_task


# ============================================================================
# GET SINGLE TASK ENDPOINT
# ============================================================================
# GET /tasks/{task_id} - Get a specific task by ID

@app.get(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    summary="Get a task by ID",
    tags=["Tasks"]
)
def get_task_endpoint(
    task_id: int,
    db: Session = Depends(get_db)
) -> TaskResponse:
    """
    Get a single task by ID
    
    Args:
        task_id (int): ID of the task to retrieve (from URL path)
        db (Session): Database session (injected by FastAPI)
    
    Returns:
        TaskResponse: The requested task
    
    Raises:
        HTTPException: 404 if task not found
    
    Explanation:
        {task_id} - Path parameter, FastAPI extracts from URL
        HTTPException - FastAPI error response
        status.HTTP_404_NOT_FOUND - Standard 404 status code
    """
    
    # Call CRUD function to get task
    db_task = crud.get_task(db=db, task_id=task_id)
    
    # If task not found, raise exception
    if not db_task:
        # HTTPException sends error response to client
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with id {task_id} not found"
        )
    
    # Return the task
    return db_task


# ============================================================================
# GET ALL TASKS ENDPOINT
# ============================================================================
# GET /tasks - Get all tasks with pagination and filtering

@app.get(
    "/tasks",
    response_model=TaskListResponse,
    summary="Get all tasks",
    tags=["Tasks"]
)
def get_all_tasks_endpoint(
    skip: int = Query(0, ge=0, description="Number of tasks to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum tasks to return"),
    completed: Optional[bool] = Query(None, description="Filter by completion status"),
    db: Session = Depends(get_db)
) -> TaskListResponse:
    """
    Get all tasks with pagination and optional filtering
    
    Args:
        skip (int): Number of tasks to skip (pagination offset)
        limit (int): Maximum tasks to return (pagination limit)
        completed (bool): Filter by completion status (optional)
        db (Session): Database session (injected)
    
    Returns:
        TaskListResponse: List of tasks with metadata
    
    Explanation:
        Query() - Validates query parameters (?skip=0&limit=10&completed=false)
        ge=0 - greater than or equal to 0
        le=100 - less than or equal to 100
        Optional[bool] - This parameter is optional
    """
    
    # Get total count of tasks
    total = crud.get_tasks_count(db=db)
    
    # Get paginated tasks
    tasks = crud.get_all_tasks(
        db=db,
        skip=skip,
        limit=limit,
        completed=completed
    )
    
    # Return response with tasks and metadata
    return TaskListResponse(
        tasks=tasks,
        total=total,
        count=len(tasks)
    )


# ============================================================================
# UPDATE TASK ENDPOINT
# ============================================================================
# PUT /tasks/{task_id} - Update a specific task

@app.put(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    summary="Update a task",
    tags=["Tasks"]
)
def update_task_endpoint(
    task_id: int,
    task_update: TaskUpdate,
    db: Session = Depends(get_db)
) -> TaskResponse:
    """
    Update a task
    
    Args:
        task_id (int): ID of task to update (from URL path)
        task_update (TaskUpdate): New task data from request body
        db (Session): Database session (injected)
    
    Returns:
        TaskResponse: The updated task
    
    Raises:
        HTTPException: 404 if task not found
    
    Explanation:
        PUT - Standard HTTP method for full/partial updates
        task_update - Request body validated by TaskUpdate schema
    """
    
    # Call CRUD function to update task
    db_task = crud.update_task(db=db, task_id=task_id, task_update=task_update)
    
    # If task not found, raise exception
    if not db_task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with id {task_id} not found"
        )
    
    # Return the updated task
    return db_task


# ============================================================================
# DELETE TASK ENDPOINT
# ============================================================================
# DELETE /tasks/{task_id} - Delete a specific task

@app.delete(
    "/tasks/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a task",
    tags=["Tasks"]
)
def delete_task_endpoint(
    task_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a task
    
    Args:
        task_id (int): ID of task to delete (from URL path)
        db (Session): Database session (injected)
    
    Raises:
        HTTPException: 404 if task not found
    
    Explanation:
        DELETE - HTTP method for deletion
        status_code=204 - No Content (standard response after successful delete)
        204 means: operation successful, no response body needed
    """
    
    # Call CRUD function to delete task
    success = crud.delete_task(db=db, task_id=task_id)
    
    # If task not found, raise exception
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with id {task_id} not found"
        )
    
    # No return value for 204 status
    # FastAPI will automatically return empty body with 204 status


# ============================================================================
# MARK TASK AS COMPLETED ENDPOINT
# ============================================================================
# PATCH /tasks/{task_id}/complete - Mark a task as completed

@app.patch(
    "/tasks/{task_id}/complete",
    response_model=TaskResponse,
    summary="Mark task as completed",
    tags=["Tasks"]
)
def complete_task_endpoint(
    task_id: int,
    db: Session = Depends(get_db)
) -> TaskResponse:
    """
    Mark a task as completed
    
    Args:
        task_id (int): ID of task to mark as completed
        db (Session): Database session (injected)
    
    Returns:
        TaskResponse: The updated task
    
    Raises:
        HTTPException: 404 if task not found
    
    Explanation:
        PATCH - Used for partial updates (unlike PUT which replaces entire object)
        This endpoint only updates the 'completed' field to True
    """
    
    # Create an update object with only completed=True
    update_data = TaskUpdate(completed=True)
    
    # Update the task
    db_task = crud.update_task(db=db, task_id=task_id, task_update=update_data)
    
    # If task not found, raise exception
    if not db_task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with id {task_id} not found"
        )
    
    # Return the updated task
    return db_task


# ============================================================================
# STARTUP EVENT
# ============================================================================
# This runs when the application starts

@app.on_event("startup")
async def startup_event():
    """
    Startup event handler
    
    Explanation:
        @app.on_event("startup") - Run this function when app starts
        Useful for initialization, loading data, etc.
    """
    print("✅ Application starting up...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔄 Alternative docs: http://localhost:8000/redoc")


# ============================================================================
# SHUTDOWN EVENT
# ============================================================================
# This runs when the application shuts down

@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler
    
    Explanation:
        @app.on_event("shutdown") - Run when app shuts down
        Useful for cleanup, closing connections, etc.
    """
    print("👋 Application shutting down...")


# ============================================================================
# HOW TO RUN THIS APPLICATION
# ============================================================================
# In terminal, run this command:
# uvicorn main:app --reload
#
# Breakdown:
# - uvicorn: The ASGI server
# - main: The module (main.py)
# - app: The FastAPI instance variable name
# - --reload: Restart on file changes (development only)
#
# Output will show:
# Uvicorn running on http://127.0.0.1:8000
# Visit http://localhost:8000/docs for interactive API documentation
```

-----

## 📝 Step-by-Step Setup & Testing

### Step 1: Create Project Directory

```bash
mkdir task-api
cd task-api
```

### Step 2: Create Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install fastapi uvicorn sqlalchemy pydantic
```

### Step 4: Create All Files

Create the five Python files with code shown above:

- `database.py`
- `models.py`
- `schemas.py`
- `crud.py`
- `main.py`

### Step 5: Run the Application

```bash
uvicorn main:app --reload
```

You should see:

```
Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

-----

## 🧪 Testing the API

### Method 1: Using Interactive API Docs

1. Go to: **http://localhost:8000/docs**
1. Try each endpoint with the “Try it out” button

### Method 2: Using cURL (Command Line)

```bash
# CREATE a task
curl -X POST "http://localhost:8000/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Learn FastAPI",
    "description": "Complete FastAPI tutorial"
  }'

# Response:
# {
#   "id": 1,
#   "title": "Learn FastAPI",
#   "description": "Complete FastAPI tutorial",
#   "completed": false,
#   "created_at": "2024-01-15T10:30:00"
# }


# GET all tasks
curl "http://localhost:8000/tasks"

# Response:
# {
#   "tasks": [...],
#   "total": 1,
#   "count": 1
# }


# GET single task
curl "http://localhost:8000/tasks/1"


# UPDATE a task
curl -X PUT "http://localhost:8000/tasks/1" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Learn FastAPI - Updated",
    "completed": true
  }'


# MARK as completed
curl -X PATCH "http://localhost:8000/tasks/1/complete"


# DELETE a task
curl -X DELETE "http://localhost:8000/tasks/1"
```

### Method 3: Using Python Requests

```python
import requests

# Create task
response = requests.post(
    "http://localhost:8000/tasks",
    json={
        "title": "Learn FastAPI",
        "description": "Complete tutorial"
    }
)
print(response.json())

# Get all tasks
response = requests.get("http://localhost:8000/tasks")
print(response.json())

# Get single task
response = requests.get("http://localhost:8000/tasks/1")
print(response.json())

# Update task
response = requests.put(
    "http://localhost:8000/tasks/1",
    json={"completed": True}
)
print(response.json())

# Delete task
response = requests.delete("http://localhost:8000/tasks/1")
print(response.status_code)  # 204
```

-----

## 🎓 Key FastAPI Concepts Explained

### 1. **Decorators**

```python
@app.get("/path")      # Register GET endpoint
@app.post("/path")     # Register POST endpoint
@app.put("/path")      # Register PUT endpoint
@app.delete("/path")   # Register DELETE endpoint
@app.patch("/path")    # Register PATCH endpoint
```

### 2. **Path Parameters** (in URL)

```python
@app.get("/tasks/{task_id}")  # {task_id} is path parameter
def get_task(task_id: int):   # Type hints validate data type
    pass
```

### 3. **Query Parameters** (after ?)

```python
# http://localhost:8000/tasks?skip=0&limit=10
@app.get("/tasks")
def get_tasks(skip: int = 0, limit: int = 10):
    pass
```

### 4. **Request Body** (JSON)

```python
@app.post("/tasks")
def create_task(task: TaskCreate):  # task comes from JSON body
    pass
```

### 5. **Response Model Validation**

```python
@app.get("/tasks", response_model=TaskListResponse)
# FastAPI validates response matches schema
```

### 6. **Dependency Injection**

```python
def get_tasks(db: Session = Depends(get_db)):
    # FastAPI calls get_db() and injects result
    pass
```

### 7. **HTTP Status Codes**

```python
@app.post("/tasks", status_code=status.HTTP_201_CREATED)
# 201 = Resource created
# 204 = No content (after delete)
# 400 = Bad request
# 404 = Not found
# 500 = Server error
```

### 8. **Error Handling**

```python
if not found:
    raise HTTPException(
        status_code=404,
        detail="Not found"
    )
```

### 9. **Validation with Pydantic**

```python
class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    # ... validates automatically
```

### 10. **Middleware**

```python
@app.middleware("http")
async def my_middleware(request, call_next):
    response = await call_next(request)
    return response
```

-----

## 🚀 Advanced Concepts to Learn Next

1. **Authentication** - Add JWT tokens for security
1. **Rate Limiting** - Prevent API abuse
1. **Background Tasks** - Run async jobs
1. **File Uploads** - Handle file uploads
1. **WebSockets** - Real-time communication
1. **Testing** - Unit tests with pytest
1. **Deployment** - Deploy to production

-----

## 📚 FastAPI Features Used in This Project

|Feature                 |Usage                |
|------------------------|---------------------|
|**Path Parameters**     |`/tasks/{task_id}`   |
|**Query Parameters**    |`?skip=0&limit=10`   |
|**Request Body**        |POST/PUT with JSON   |
|**Response Models**     |Pydantic validation  |
|**Status Codes**        |201, 204, 404, etc.  |
|**Dependency Injection**|Database sessions    |
|**CORS Middleware**     |Cross-origin requests|
|**Custom Middleware**   |Request processing   |
|**Error Handling**      |HTTPException        |
|**Validation**          |Pydantic with Field()|
|**Type Hints**          |Full type safety     |
|**Auto Documentation**  |Swagger & ReDoc      |
|**Events**              |Startup/Shutdown     |
|**Async Support**       |Concurrent requests  |

-----

## 💡 Common Mistakes to Avoid

```python
# ❌ WRONG: Not using Depends() for database
def get_tasks(db: Session):
    pass

# ✅ RIGHT: Use Depends() for dependency injection
def get_tasks(db: Session = Depends(get_db)):
    pass


# ❌ WRONG: Not validating input
def create_task(data: dict):
    pass

# ✅ RIGHT: Use Pydantic models for validation
def create_task(data: TaskCreate):
    pass


# ❌ WRONG: Forgetting to commit database changes
db.add(task)
# Missing: db.commit()

# ✅ RIGHT: Always commit after modifications
db.add(task)
db.commit()
db.refresh(task)


# ❌ WRONG: Not handling errors
def get_task(task_id: int):
    task = crud.get_task(db, task_id)
    return task  # Crashes if None

# ✅ RIGHT: Check and raise HTTPException
def get_task(task_id: int):
    task = crud.get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Not found")
    return task
```

-----

## 📖 Summary

You now have a **complete, production-ready FastAPI project** with:

✅ **CRUD Operations** - Create, Read, Update, Delete  
✅ **Database Integration** - SQLAlchemy + SQLite  
✅ **Validation** - Pydantic models  
✅ **Error Handling** - Proper HTTP exceptions  
✅ **Middleware** - CORS + custom logging  
✅ **Type Safety** - Full type hints  
✅ **Auto Documentation** - Interactive API docs  
✅ **Async Support** - Concurrent request handling

Every line has detailed explanations so you understand **what, why, and how** each part works!

-----

## 🎯 Practice Challenges

Try implementing these features:

1. **Search Tasks** - Add `/tasks/search?q=keyword` endpoint
1. **Sorting** - Add `/tasks?sort=created_at&order=asc`
1. **Categories** - Add task categories/tags
1. **User Accounts** - Add users and task ownership
1. **Authentication** - Add JWT token security
1. **Notifications** - Email when task is due
1. **Comments** - Add comments to tasks
1. **Sharing** - Share tasks with other users

-----

Happy Learning! 🚀