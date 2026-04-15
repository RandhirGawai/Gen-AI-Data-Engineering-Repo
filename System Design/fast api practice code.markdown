# FastAPI Complete Learning Guide - Build a Task Management API
# FastAPI Interview Practice Questions & Solutions

## 

# FastAPI Practice Questions & Solutions

## Overview

This guide contains **10 progressive FastAPI practice problems** with complete solutions and detailed line-by-line explanations. We’ll use in-memory data structures (lists and dicts) for storage.

**Prerequisites:** Python 3.7+, FastAPI, Uvicorn

```bash
pip install fastapi uvicorn
```

-----

## **Problem 1: Basic GET Request - Get All Users**

### Question

Create a FastAPI endpoint that:

- Returns a list of all users
- Each user has: `id`, `name`, `email`
- Use a list to store users
- GET endpoint at `/users`

### Solution

```python
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel

app = FastAPI()

# 1. Define the User model using Pydantic
class User(BaseModel):
    id: int
    name: str
    email: str

# 2. Store users in memory (list of dictionaries)
users_db = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
]

# 3. GET endpoint to fetch all users
@app.get("/users", response_model=List[User])
def get_all_users():
    return users_db

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Line-by-Line Explanation

```python
from fastapi import FastAPI
```

- Imports the main FastAPI class to create the application instance.

```python
from typing import List
```

- Imports `List` type hint for type annotations (tells Python and IDE we’re returning a list).

```python
from pydantic import BaseModel
```

- Imports `BaseModel` from Pydantic for data validation and serialization.

```python
app = FastAPI()
```

- Creates a FastAPI application instance. This is the core of our API.

```python
class User(BaseModel):
    id: int
    name: str
    email: str
```

- Defines a **Pydantic model** for request/response validation.
- `BaseModel` automatically validates incoming data against these types.
- If data doesn’t match types, FastAPI returns a validation error automatically.

```python
users_db = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    ...
]
```

- Creates an in-memory database as a Python list containing dictionary objects.

```python
@app.get("/users", response_model=List[User])
```

- `@app.get()` → Decorator that creates a GET HTTP endpoint.
- `"/users"` → The URL path for this endpoint.
- `response_model=List[User]` → Tells FastAPI to validate/serialize response as a list of User objects.

```python
def get_all_users():
    return users_db
```

- Function that handles the endpoint logic.
- Returns the users list directly.

### Test It

```bash
# Start server
python app.py

# In another terminal, test the endpoint
curl http://localhost:8000/users

# Or visit in browser
http://localhost:8000/users
```

-----

## **Problem 2: GET Request with Path Parameter - Get User by ID**

### Question

Create an endpoint that:

- Gets a single user by their `id`
- Returns 404 if user not found
- GET endpoint at `/users/{user_id}`

### Solution

```python
from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    id: int
    name: str
    email: str

users_db = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
]

# 1. GET endpoint with path parameter
@app.get("/users/{user_id}", response_model=User)
def get_user(user_id: int):
    # 2. Loop through users_db to find matching user
    for user in users_db:
        if user["id"] == user_id:
            return user
    
    # 3. If user not found, raise HTTP 404 exception
    raise HTTPException(status_code=404, detail="User not found")
```

### Line-by-Line Explanation

```python
@app.get("/users/{user_id}", response_model=User)
```

- `{user_id}` → Path parameter that will be extracted from the URL.
- FastAPI automatically extracts this from the URL and passes it to the function.

```python
def get_user(user_id: int):
```

- Function receives `user_id` as a parameter.
- Type hint `int` tells FastAPI to:
  - Convert the string from URL to integer
  - Return 422 validation error if conversion fails

```python
for user in users_db:
    if user["id"] == user_id:
        return user
```

- Iterates through the users list.
- Checks if user’s id matches the requested user_id.
- Returns the matching user.

```python
raise HTTPException(status_code=404, detail="User not found")
```

- If loop completes without finding user, raises HTTP 404 exception.
- `detail` parameter provides error message in response body.

### Test It

```bash
curl http://localhost:8000/users/1
# Returns: {"id": 1, "name": "Alice", "email": "alice@example.com"}

curl http://localhost:8000/users/999
# Returns: {"detail": "User not found"} with 404 status
```

-----

## **Problem 3: POST Request - Create a New User**

### Question

Create an endpoint that:

- Accepts user data (name, email)
- Generates a new ID automatically
- Adds user to the database
- Returns the created user
- POST endpoint at `/users`

### Solution

```python
from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    id: int
    name: str
    email: str

class UserCreate(BaseModel):  # 1. Request model without ID
    name: str
    email: str

users_db = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
]

# 2. Helper function to get next available ID
def get_next_user_id():
    if not users_db:
        return 1
    return max(user["id"] for user in users_db) + 1

# 3. POST endpoint to create user
@app.post("/users", response_model=User, status_code=201)
def create_user(user: UserCreate):
    # 4. Generate new ID
    new_id = get_next_user_id()
    
    # 5. Create new user dictionary
    new_user = {
        "id": new_id,
        "name": user.name,
        "email": user.email
    }
    
    # 6. Add to database
    users_db.append(new_user)
    
    # 7. Return the created user
    return new_user
```

### Line-by-Line Explanation

```python
class UserCreate(BaseModel):
    name: str
    email: str
```

- Separate model for **request body** (without ID, since server generates it).
- This is different from `User` response model which includes ID.

```python
def get_next_user_id():
    if not users_db:
        return 1
    return max(user["id"] for user in users_db) + 1
```

- Helper function to auto-generate the next ID.
- `if not users_db:` → Checks if list is empty, return 1 if it is.
- `max(user["id"] for user in users_db) + 1` → Finds the maximum ID and adds 1.

```python
@app.post("/users", response_model=User, status_code=201)
```

- `@app.post()` → Creates a POST endpoint (for creating resources).
- `status_code=201` → Returns HTTP 201 (Created) status instead of default 200.

```python
def create_user(user: UserCreate):
```

- `user` parameter is automatically validated against `UserCreate` model.
- FastAPI validates that JSON body contains `name` and `email` fields.

```python
new_user = {
    "id": new_id,
    "name": user.name,
    "email": user.email
}
```

- Creates a dictionary from the Pydantic model data.
- `user.name` accesses the name field from the UserCreate object.

```python
users_db.append(new_user)
```

- Adds the new user dictionary to the list.

### Test It

```bash
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"name": "David", "email": "david@example.com"}'

# Returns: {"id": 4, "name": "David", "email": "david@example.com"} with status 201
```

-----

## **Problem 4: PUT Request - Update User**

### Question

Create an endpoint that:

- Updates an existing user by ID
- Accepts updated name and email
- Returns 404 if user not found
- PUT endpoint at `/users/{user_id}`

### Solution

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    id: int
    name: str
    email: str

class UserUpdate(BaseModel):  # 1. Model for update (no ID)
    name: str
    email: str

users_db = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
]

# 2. PUT endpoint to update user
@app.put("/users/{user_id}", response_model=User)
def update_user(user_id: int, user_update: UserUpdate):
    # 3. Find the user in database
    for i, user in enumerate(users_db):
        if user["id"] == user_id:
            # 4. Update the user data
            users_db[i]["name"] = user_update.name
            users_db[i]["email"] = user_update.email
            # 5. Return updated user
            return users_db[i]
    
    # 6. If not found, raise 404
    raise HTTPException(status_code=404, detail="User not found")
```

### Line-by-Line Explanation

```python
class UserUpdate(BaseModel):
    name: str
    email: str
```

- Separate model for update requests (client provides both name and email).
- Doesn’t include ID since that’s immutable.

```python
for i, user in enumerate(users_db):
```

- `enumerate()` gives both the **index** (`i`) and the **value** (`user`).
- We need the index to update the user at that position in the list.

```python
if user["id"] == user_id:
```

- Checks if current user’s ID matches the requested user_id.

```python
users_db[i]["name"] = user_update.name
users_db[i]["email"] = user_update.email
```

- Updates the values at the found index.
- `user_update.name` accesses the name from the request model.

```python
return users_db[i]
```

- Returns the updated user from the list.

### Test It

```bash
curl -X PUT http://localhost:8000/users/1 \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice Smith", "email": "alice.smith@example.com"}'

# Returns: {"id": 1, "name": "Alice Smith", "email": "alice.smith@example.com"}
```

-----

## **Problem 5: DELETE Request - Delete User**

### Question

Create an endpoint that:

- Deletes a user by ID
- Returns 404 if user not found
- Returns a confirmation message
- DELETE endpoint at `/users/{user_id}`

### Solution

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    id: int
    name: str
    email: str

users_db = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
]

# 1. DELETE endpoint
@app.delete("/users/{user_id}")
def delete_user(user_id: int):
    # 2. Iterate with enumerate to get index
    for i, user in enumerate(users_db):
        if user["id"] == user_id:
            # 3. Remove user at index i
            deleted_user = users_db.pop(i)
            # 4. Return confirmation with deleted user data
            return {
                "message": f"User {user_id} deleted successfully",
                "deleted_user": deleted_user
            }
    
    # 5. If not found, raise 404
    raise HTTPException(status_code=404, detail="User not found")
```

### Line-by-Line Explanation

```python
@app.delete("/users/{user_id}")
```

- `@app.delete()` → Creates a DELETE HTTP endpoint.
- Used to delete resources.

```python
deleted_user = users_db.pop(i)
```

- `pop(i)` → Removes the element at index `i` and returns it.
- Unlike `del users_db[i]`, pop returns the removed item.

```python
return {
    "message": f"User {user_id} deleted successfully",
    "deleted_user": deleted_user
}
```

- Returns a dictionary with confirmation message.
- Uses f-string to include the user_id in the message.

### Test It

```bash
curl -X DELETE http://localhost:8000/users/1

# Returns: 
# {
#   "message": "User 1 deleted successfully",
#   "deleted_user": {"id": 1, "name": "Alice", "email": "alice@example.com"}
# }
```

-----

## **Problem 6: Query Parameters - Filter and Pagination**

### Question

Create an endpoint that:

- Fetches users with optional filtering by name
- Supports pagination (skip and limit)
- GET endpoint at `/users`

### Solution

```python
from fastapi import FastAPI
from typing import List, Optional
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    id: int
    name: str
    email: str

users_db = [
    {"id": 1, "name": "Alice Johnson", "email": "alice@example.com"},
    {"id": 2, "name": "Bob Smith", "email": "bob@example.com"},
    {"id": 3, "name": "Charlie Brown", "email": "charlie@example.com"},
    {"id": 4, "name": "Diana Prince", "email": "diana@example.com"},
    {"id": 5, "name": "Eve Wilson", "email": "eve@example.com"}
]

# 1. GET endpoint with query parameters
@app.get("/users", response_model=List[User])
def get_users(
    # 2. Optional query parameters
    name: Optional[str] = None,  # Filter by name (contains)
    skip: int = 0,                # Number of records to skip
    limit: int = 10               # Maximum records to return
):
    # 3. Start with all users
    result = users_db
    
    # 4. Filter by name if provided
    if name:
        result = [user for user in result if name.lower() in user["name"].lower()]
    
    # 5. Apply pagination
    result = result[skip : skip + limit]
    
    # 6. Return filtered and paginated results
    return result
```

### Line-by-Line Explanation

```python
def get_users(
    name: Optional[str] = None,
    skip: int = 0,
    limit: int = 10
):
```

- `Optional[str]` → Parameter is either a string or None.
- `= None` → Default value is None (parameter is optional in URL).
- `skip: int = 0` → How many records to skip, defaults to 0.
- `limit: int = 10` → How many records to return, defaults to 10.

```python
if name:
    result = [user for user in result if name.lower() in user["name"].lower()]
```

- **List comprehension** that filters users.
- `name.lower()` → Converts to lowercase for case-insensitive search.
- `in user["name"].lower()` → Checks if name contains the search string (substring match).

```python
result = result[skip : skip + limit]
```

- Python list slicing for pagination.
- `skip` → Starting index
- `skip + limit` → Ending index (not inclusive)
- Example: If skip=2 and limit=3, returns items at index 2, 3, 4.

### Test It

```bash
# Get all users with default pagination
curl "http://localhost:8000/users"

# Get users with name containing "john"
curl "http://localhost:8000/users?name=john"

# Get 2 users, skip first 2
curl "http://localhost:8000/users?skip=2&limit=2"

# Combine filters and pagination
curl "http://localhost:8000/users?name=alice&skip=0&limit=5"
```

-----

## **Problem 7: Request Body Validation - Multiple Fields**

### Question

Create an endpoint that:

- Creates a product with validation
- Name must be at least 3 characters
- Price must be greater than 0
- Stock must be >= 0
- POST endpoint at `/products`

### Solution

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

app = FastAPI()

# 1. Define Product model with validation
class Product(BaseModel):
    id: int
    name: str = Field(..., min_length=3, max_length=100)  # name is required, min 3 chars
    price: float = Field(..., gt=0)                        # price must be > 0
    stock: int = Field(..., ge=0)                          # stock must be >= 0
    description: str = Field(None, max_length=500)         # optional, max 500 chars

class ProductCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=100)
    price: float = Field(..., gt=0)
    stock: int = Field(..., ge=0)
    description: str = Field(None, max_length=500)

products_db = [
    {"id": 1, "name": "Laptop", "price": 999.99, "stock": 5, "description": "Gaming laptop"},
    {"id": 2, "name": "Mouse", "price": 29.99, "stock": 50, "description": "Wireless mouse"}
]

def get_next_product_id():
    if not products_db:
        return 1
    return max(p["id"] for p in products_db) + 1

# 2. POST endpoint
@app.post("/products", response_model=Product, status_code=201)
def create_product(product: ProductCreate):
    # 3. Create new product with generated ID
    new_product = {
        "id": get_next_product_id(),
        "name": product.name,
        "price": product.price,
        "stock": product.stock,
        "description": product.description
    }
    
    # 4. Add to database
    products_db.append(new_product)
    
    # 5. Return created product
    return new_product
```

### Line-by-Line Explanation

```python
name: str = Field(..., min_length=3, max_length=100)
```

- `Field()` → Function to add validation rules to fields.
- `...` → Means this field is required (cannot be omitted).
- `min_length=3` → String must be at least 3 characters.
- `max_length=100` → String cannot exceed 100 characters.

```python
price: float = Field(..., gt=0)
```

- `gt=0` → “greater than” - price must be strictly greater than 0.
- Other options: `ge=0` (>=), `lt=X` (<), `le=X` (<=)

```python
stock: int = Field(..., ge=0)
```

- `ge=0` → “greater than or equal” - stock can be 0 or more.

```python
description: str = Field(None, max_length=500)
```

- `None` as default → This field is **optional**.
- Can be omitted from request body.
- `max_length=500` → If provided, max 500 characters.

### Test It

```bash
# Valid request
curl -X POST http://localhost:8000/products \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Keyboard",
    "price": 79.99,
    "stock": 20,
    "description": "Mechanical keyboard"
  }'
# Returns: 201 Created with product data

# Invalid request - name too short
curl -X POST http://localhost:8000/products \
  -H "Content-Type: application/json" \
  -d '{"name": "PC", "price": 79.99, "stock": 20}'
# Returns: 422 Validation Error - "String should have at least 3 characters"

# Invalid request - negative price
curl -X POST http://localhost:8000/products \
  -H "Content-Type: application/json" \
  -d '{"name": "Keyboard", "price": -10, "stock": 20}'
# Returns: 422 Validation Error - "Input should be greater than 0"
```

-----

## **Problem 8: Multiple Response Models - Success and Error Cases**

### Question

Create endpoints for a task management system:

- Create task (title, description, completed)
- Get all tasks
- Mark task as completed
- Include proper status codes and error handling
- GET, POST, PATCH endpoints

### Solution

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from enum import Enum

app = FastAPI()

# 1. Define Task model
class Task(BaseModel):
    id: int
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(None, max_length=1000)
    completed: bool = False

# 2. Model for creating task (no id, completed defaults to False)
class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(None, max_length=1000)

# 3. Model for updating task completion status
class TaskUpdate(BaseModel):
    completed: bool

# 4. Error response model (optional but good practice)
class ErrorResponse(BaseModel):
    detail: str

tasks_db = [
    {"id": 1, "title": "Learn FastAPI", "description": "Complete FastAPI tutorial", "completed": False},
    {"id": 2, "title": "Build API", "description": "Create REST API", "completed": True}
]

def get_next_task_id():
    if not tasks_db:
        return 1
    return max(t["id"] for t in tasks_db) + 1

# 5. GET all tasks
@app.get("/tasks", response_model=List[Task])
def get_all_tasks():
    return tasks_db

# 6. GET single task by ID
@app.get("/tasks/{task_id}", response_model=Task)
def get_task(task_id: int):
    for task in tasks_db:
        if task["id"] == task_id:
            return task
    raise HTTPException(status_code=404, detail="Task not found")

# 7. POST - Create new task
@app.post("/tasks", response_model=Task, status_code=201)
def create_task(task_create: TaskCreate):
    new_task = {
        "id": get_next_task_id(),
        "title": task_create.title,
        "description": task_create.description,
        "completed": False
    }
    tasks_db.append(new_task)
    return new_task

# 8. PATCH - Update task completion status
@app.patch("/tasks/{task_id}", response_model=Task)
def update_task(task_id: int, task_update: TaskUpdate):
    for i, task in enumerate(tasks_db):
        if task["id"] == task_id:
            tasks_db[i]["completed"] = task_update.completed
            return tasks_db[i]
    raise HTTPException(status_code=404, detail="Task not found")

# 9. DELETE task
@app.delete("/tasks/{task_id}")
def delete_task(task_id: int):
    for i, task in enumerate(tasks_db):
        if task["id"] == task_id:
            deleted_task = tasks_db.pop(i)
            return {"message": f"Task {task_id} deleted", "task": deleted_task}
    raise HTTPException(status_code=404, detail="Task not found")
```

### Line-by-Line Explanation

```python
class TaskCreate(BaseModel):
    title: str = Field(...)
    description: str = Field(None, max_length=1000)
```

- Separate models for different operations.
- `TaskCreate` for POST (no ID, completed is implicit False).

```python
class TaskUpdate(BaseModel):
    completed: bool
```

- Minimal model for PATCH - only allows updating the `completed` field.

```python
@app.patch("/tasks/{task_id}", response_model=Task)
```

- `@app.patch()` → HTTP PATCH method for partial updates.
- PATCH updates only specified fields (vs PUT which replaces entire resource).

```python
tasks_db[i]["completed"] = task_update.completed
return tasks_db[i]
```

- Updates only the `completed` field.
- Returns the full updated task.

### Test It

```bash
# Create task
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"title": "Test task", "description": "Testing"}'

# Get all tasks
curl http://localhost:8000/tasks

# Get single task
curl http://localhost:8000/tasks/1

# Update task completion
curl -X PATCH http://localhost:8000/tasks/1 \
  -H "Content-Type: application/json" \
  -d '{"completed": true}'

# Delete task
curl -X DELETE http://localhost:8000/tasks/1
```

-----

## **Problem 9: Working with Nested Models**

### Question

Create an endpoint that:

- Creates a blog post with comments
- Blog post has: title, content, author (nested)
- Author has: name, email
- Comments are a list of comment objects
- POST and GET endpoints

### Solution

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

app = FastAPI()

# 1. Define nested Author model
class Author(BaseModel):
    name: str = Field(..., min_length=1)
    email: str = Field(..., min_length=5)

# 2. Define Comment model
class Comment(BaseModel):
    id: int
    text: str = Field(..., min_length=1, max_length=500)
    author_name: str
    created_at: str

# 3. Define BlogPost model with nested structures
class BlogPost(BaseModel):
    id: int
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    author: Author  # Nested object
    comments: List[Comment] = []  # List of nested objects
    created_at: str

# 4. Model for creating blog post (without id and created_at)
class BlogPostCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    author: Author  # Still include nested author

# 5. Model for adding comments
class CommentCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    author_name: str

posts_db = [
    {
        "id": 1,
        "title": "Getting Started with FastAPI",
        "content": "FastAPI is a modern web framework...",
        "author": {"name": "John Doe", "email": "john@example.com"},
        "comments": [
            {"id": 1, "text": "Great post!", "author_name": "Jane", "created_at": "2024-01-01"}
        ],
        "created_at": "2024-01-01"
    }
]

def get_next_post_id():
    if not posts_db:
        return 1
    return max(p["id"] for p in posts_db) + 1

def get_next_comment_id(post_id: int):
    post = next((p for p in posts_db if p["id"] == post_id), None)
    if not post or not post["comments"]:
        return 1
    return max(c["id"] for c in post["comments"]) + 1

# 6. POST - Create new blog post
@app.post("/posts", response_model=BlogPost, status_code=201)
def create_post(post_create: BlogPostCreate):
    new_post = {
        "id": get_next_post_id(),
        "title": post_create.title,
        "content": post_create.content,
        "author": {
            "name": post_create.author.name,
            "email": post_create.author.email
        },
        "comments": [],
        "created_at": datetime.now().isoformat()
    }
    posts_db.append(new_post)
    return new_post

# 7. GET all posts
@app.get("/posts", response_model=List[BlogPost])
def get_all_posts():
    return posts_db

# 8. GET single post by ID
@app.get("/posts/{post_id}", response_model=BlogPost)
def get_post(post_id: int):
    for post in posts_db:
        if post["id"] == post_id:
            return post
    raise HTTPException(status_code=404, detail="Post not found")

# 9. POST - Add comment to post
@app.post("/posts/{post_id}/comments", response_model=BlogPost)
def add_comment(post_id: int, comment_create: CommentCreate):
    for post in posts_db:
        if post["id"] == post_id:
            # Create new comment
            new_comment = {
                "id": get_next_comment_id(post_id),
                "text": comment_create.text,
                "author_name": comment_create.author_name,
                "created_at": datetime.now().isoformat()
            }
            # Add to comments list
            post["comments"].append(new_comment)
            # Return updated post
            return post
    
    raise HTTPException(status_code=404, detail="Post not found")
```

### Line-by-Line Explanation

```python
class Author(BaseModel):
    name: str = Field(..., min_length=1)
    email: str = Field(..., min_length=5)
```

- Defines a **nested model** - Author is a model within BlogPost.

```python
class BlogPost(BaseModel):
    ...
    author: Author  # Nested object
    comments: List[Comment] = []  # List of nested objects
```

- `author: Author` → Expects an Author object (not just a string).
- `comments: List[Comment] = []` → List of Comment objects, defaults to empty list.

```python
class BlogPostCreate(BaseModel):
    ...
    author: Author  # Still include nested author
```

- Request model also includes the nested Author object.
- Client must provide author data in the request.

```python
"author": {
    "name": post_create.author.name,
    "email": post_create.author.email
}
```

- Accessing nested object fields using dot notation.
- `post_create.author.name` accesses the name field of the Author object.

```python
new_comment = {
    "id": get_next_comment_id(post_id),
    "text": comment_create.text,
    "author_name": comment_create.author_name,
    "created_at": datetime.now().isoformat()
}
post["comments"].append(new_comment)
```

- Creates a comment dictionary.
- Appends to the `comments` list of the post.

### Test It

```bash
# Create blog post
curl -X POST http://localhost:8000/posts \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Python Best Practices",
    "content": "Always write clean code...",
    "author": {
      "name": "Alice Smith",
      "email": "alice@example.com"
    }
  }'

# Get all posts
curl http://localhost:8000/posts

# Add comment to post
curl -X POST http://localhost:8000/posts/1/comments \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Excellent article!",
    "author_name": "Bob Johnson"
  }'
```

-----

## **Problem 10: Combining Everything - E-Commerce API**

### Question

Build a mini e-commerce API with:

- Products CRUD operations
- Shopping cart (per session/user)
- Add/remove items from cart
- Calculate total price
- Place order (converts cart to order)

### Solution

```python
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

app = FastAPI()

# ===== MODELS =====

class Product(BaseModel):
    id: int
    name: str = Field(..., min_length=1)
    price: float = Field(..., gt=0)
    stock: int = Field(..., ge=0)

class ProductCreate(BaseModel):
    name: str = Field(..., min_length=1)
    price: float = Field(..., gt=0)
    stock: int = Field(..., ge=0)

class CartItem(BaseModel):
    product_id: int
    quantity: int = Field(..., gt=0)
    name: str
    price: float
    subtotal: float

class Cart(BaseModel):
    items: List[CartItem]
    total_price: float

class OrderStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class Order(BaseModel):
    id: int
    items: List[CartItem]
    total_price: float
    status: OrderStatus
    created_at: str

# ===== DATABASES =====

products_db = [
    {"id": 1, "name": "Laptop", "price": 999.99, "stock": 5},
    {"id": 2, "name": "Mouse", "price": 29.99, "stock": 50},
    {"id": 3, "name": "Keyboard", "price": 79.99, "stock": 30}
]

# 1. Cart storage - keyed by user_id or session_id
carts_db: Dict[str, List[Dict]] = {}

# 2. Orders storage
orders_db = []

# ===== HELPER FUNCTIONS =====

def get_next_product_id():
    if not products_db:
        return 1
    return max(p["id"] for p in products_db) + 1

def get_next_order_id():
    if not orders_db:
        return 1
    return max(o["id"] for o in orders_db) + 1

def find_product(product_id: int):
    """3. Find product by ID, raise 404 if not found"""
    for product in products_db:
        if product["id"] == product_id:
            return product
    raise HTTPException(status_code=404, detail="Product not found")

def get_or_create_cart(user_id: str):
    """4. Get user's cart or create if doesn't exist"""
    if user_id not in carts_db:
        carts_db[user_id] = []
    return carts_db[user_id]

def calculate_cart_total(cart_items):
    """5. Calculate total price of cart"""
    return sum(item["subtotal"] for item in cart_items)

# ===== PRODUCT ENDPOINTS =====

@app.post("/products", response_model=Product, status_code=201)
def create_product(product_create: ProductCreate):
    new_product = {
        "id": get_next_product_id(),
        "name": product_create.name,
        "price": product_create.price,
        "stock": product_create.stock
    }
    products_db.append(new_product)
    return new_product

@app.get("/products", response_model=List[Product])
def get_products(skip: int = 0, limit: int = 10):
    """6. Get all products with pagination"""
    return products_db[skip : skip + limit]

@app.get("/products/{product_id}", response_model=Product)
def get_product(product_id: int):
    return find_product(product_id)

# ===== CART ENDPOINTS =====

@app.post("/cart/add")
def add_to_cart(
    user_id: str,
    product_id: int,
    quantity: int = Query(..., gt=0)
):
    """7. Add item to cart with quantity"""
    # Find product
    product = find_product(product_id)
    
    # Check stock
    if product["stock"] < quantity:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough stock. Available: {product['stock']}"
        )
    
    # Get user's cart
    cart = get_or_create_cart(user_id)
    
    # Check if product already in cart
    for item in cart:
        if item["product_id"] == product_id:
            # Update quantity
            item["quantity"] += quantity
            item["subtotal"] = item["quantity"] * item["price"]
            return {"message": "Item quantity updated"}
    
    # Add new item to cart
    new_item = {
        "product_id": product_id,
        "quantity": quantity,
        "name": product["name"],
        "price": product["price"],
        "subtotal": quantity * product["price"]
    }
    cart.append(new_item)
    
    return {"message": "Item added to cart"}

@app.get("/cart")
def get_cart(user_id: str):
    """8. Get user's cart with total"""
    cart_items = get_or_create_cart(user_id)
    return Cart(
        items=cart_items,
        total_price=calculate_cart_total(cart_items)
    )

@app.delete("/cart/remove")
def remove_from_cart(user_id: str, product_id: int):
    """9. Remove item from cart"""
    cart = get_or_create_cart(user_id)
    
    # Find and remove item
    for i, item in enumerate(cart):
        if item["product_id"] == product_id:
            removed_item = cart.pop(i)
            return {
                "message": f"Item {product_id} removed from cart",
                "removed_item": removed_item
            }
    
    raise HTTPException(status_code=404, detail="Item not in cart")

@app.delete("/cart/clear")
def clear_cart(user_id: str):
    """10. Clear entire cart"""
    if user_id in carts_db:
        carts_db[user_id] = []
    return {"message": "Cart cleared"}

# ===== ORDER ENDPOINTS =====

@app.post("/orders", response_model=Order, status_code=201)
def create_order(user_id: str):
    """11. Convert cart to order"""
    cart = get_or_create_cart(user_id)
    
    # Check cart is not empty
    if not cart:
        raise HTTPException(status_code=400, detail="Cart is empty")
    
    # Create order
    new_order = {
        "id": get_next_order_id(),
        "items": cart.copy(),  # Copy cart items
        "total_price": calculate_cart_total(cart),
        "status": "completed",
        "created_at": datetime.now().isoformat()
    }
    
    # Add to orders database
    orders_db.append(new_order)
    
    # Clear the user's cart
    carts_db[user_id] = []
    
    return new_order

@app.get("/orders", response_model=List[Order])
def get_orders(user_id: Optional[str] = None):
    """12. Get all orders (optionally filtered by user)"""
    if user_id:
        # Filter orders by user (in real app, would track user_id in order)
        return orders_db
    return orders_db

@app.get("/orders/{order_id}", response_model=Order)
def get_order(order_id: int):
    """13. Get single order by ID"""
    for order in orders_db:
        if order["id"] == order_id:
            return order
    raise HTTPException(status_code=404, detail="Order not found")
```

### Line-by-Line Explanation - Key Concepts

```python
carts_db: Dict[str, List[Dict]] = {}
```

- **Type hint:** `Dict[str, List[Dict]]`
- Key is user_id (string), value is list of cart items (list of dicts).
- Stores multiple carts, one per user.

```python
if user_id not in carts_db:
    carts_db[user_id] = []
return carts_db[user_id]
```

- Checks if user has a cart.
- Creates empty cart if doesn’t exist.
- Returns the user’s cart.

```python
def calculate_cart_total(cart_items):
    return sum(item["subtotal"] for item in cart_items)
```

- **Generator expression:** Iterates through items and sums subtotals.
- Efficient way to calculate totals.

```python
@app.post("/cart/add")
def add_to_cart(
    user_id: str,
    product_id: int,
    quantity: int = Query(..., gt=0)
):
```

- `user_id` is a query parameter (not in path).
- `Query(...)` makes it required query parameter.
- `gt=0` validates quantity is positive.

```python
for item in cart:
    if item["product_id"] == product_id:
        item["quantity"] += quantity
        item["subtotal"] = item["quantity"] * item["price"]
        return {"message": "Item quantity updated"}
```

- Checks if product already in cart.
- If yes, updates quantity and subtotal.
- Early return prevents adding duplicate.

```python
cart = get_or_create_cart(user_id)
new_order = {
    ...
    "items": cart.copy(),  # Copy cart items
    ...
}
carts_db[user_id] = []  # Clear the cart
```

- `cart.copy()` creates a **shallow copy** of the list.
- This preserves order items without referencing the original cart.

### Test It

```bash
# Create products
curl -X POST http://localhost:8000/products \
  -H "Content-Type: application/json" \
  -d '{"name": "Monitor", "price": 299.99, "stock": 10}'

# Add item to cart (user1)
curl -X POST "http://localhost:8000/cart/add?user_id=user1&product_id=1&quantity=2"

# Add another item
curl -X POST "http://localhost:8000/cart/add?user_id=user1&product_id=2&quantity=1"

# View cart
curl "http://localhost:8000/cart?user_id=user1"

# Remove item from cart
curl -X DELETE "http://localhost:8000/cart/remove?user_id=user1&product_id=2"

# Place order
curl -X POST "http://localhost:8000/orders?user_id=user1"

# Get order
curl "http://localhost:8000/orders/1"
```

-----

## Summary Table

|Problem|Concept          |Key Endpoints              |HTTP Methods                 |
|-------|-----------------|---------------------------|-----------------------------|
|1      |Basic GET        |`/users`                   |GET                          |
|2      |Path Parameters  |`/users/{id}`              |GET                          |
|3      |POST Request     |`/users`                   |POST                         |
|4      |PUT Request      |`/users/{id}`              |PUT                          |
|5      |DELETE Request   |`/users/{id}`              |DELETE                       |
|6      |Query Parameters |`/users?name=X&skip=Y`     |GET                          |
|7      |Validation       |`/products`                |POST                         |
|8      |Multiple Models  |`/tasks`                   |GET, POST, PATCH, DELETE     |
|9      |Nested Models    |`/posts`                   |GET, POST, PATCH             |
|10     |Full CRUD + Logic|`/products, /cart, /orders`|GET, POST, PUT, PATCH, DELETE|

-----

## Running the Code

### Save as `app.py`

```python
# Copy any solution code above
```

### Install Dependencies

```bash
pip install fastapi uvicorn
```

### Run Server

```bash
python app.py

# Or
uvicorn app:app --reload
```

### Access Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

-----

## Key Takeaways

✅ **Pydantic Models** → Type validation & serialization  
✅ **HTTP Methods** → GET (read), POST (create), PUT (replace), PATCH (update), DELETE (remove)  
✅ **Path Parameters** → `/users/{user_id}` - extracted from URL  
✅ **Query Parameters** → `/users?name=X` - optional filters  
✅ **Request Body** → JSON data in POST/PUT/PATCH  
✅ **Status Codes** → 200 (OK), 201 (Created), 400 (Bad Request), 404 (Not Found), 422 (Validation Error)  
✅ **Error Handling** → HTTPException for proper HTTP responses  
✅ **In-Memory Storage** → Lists and dicts for practice (use database in production)

Happy learning! 🚀





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