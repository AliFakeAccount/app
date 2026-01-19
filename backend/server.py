from fastapi import FastAPI, APIRouter, HTTPException, Response, Request, Header
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
from emergentintegrations.llm.chat import LlmChat, UserMessage
import httpx

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# ============ PYDANTIC MODELS ============

# User Models
class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    role: str = "user"
    created_at: datetime

class UserSession(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    session_token: str
    expires_at: datetime
    created_at: datetime

# Lead Models
class LeadCreate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None
    source: str = "manual"  # manual, website, email, call, ad
    status: str = "new"  # new, contacted, qualified, unqualified
    score: int = 0  # AI-generated lead score 0-100
    notes: Optional[str] = None

class Lead(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lead_id: str = Field(default_factory=lambda: f"lead_{uuid.uuid4().hex[:12]}")
    user_id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None
    source: str = "manual"
    status: str = "new"
    score: int = 0
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Contact Models
class ContactCreate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    company_id: Optional[str] = None
    title: Optional[str] = None
    linkedin: Optional[str] = None
    notes: Optional[str] = None

class Contact(BaseModel):
    model_config = ConfigDict(extra="ignore")
    contact_id: str = Field(default_factory=lambda: f"contact_{uuid.uuid4().hex[:12]}")
    user_id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    company_id: Optional[str] = None
    title: Optional[str] = None
    linkedin: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Company Models
class CompanyCreate(BaseModel):
    name: str
    industry: Optional[str] = None
    website: Optional[str] = None
    size: Optional[str] = None  # 1-10, 11-50, 51-200, 201-500, 500+
    revenue: Optional[str] = None
    notes: Optional[str] = None

class Company(BaseModel):
    model_config = ConfigDict(extra="ignore")
    company_id: str = Field(default_factory=lambda: f"company_{uuid.uuid4().hex[:12]}")
    user_id: str
    name: str
    industry: Optional[str] = None
    website: Optional[str] = None
    size: Optional[str] = None
    revenue: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Deal Models
class DealCreate(BaseModel):
    title: str
    contact_id: Optional[str] = None
    company_id: Optional[str] = None
    value: float = 0
    stage: str = "lead"  # lead, contacted, proposal, negotiation, closed_won, closed_lost
    probability: int = 10  # AI-generated probability 0-100
    expected_close_date: Optional[datetime] = None
    notes: Optional[str] = None

class Deal(BaseModel):
    model_config = ConfigDict(extra="ignore")
    deal_id: str = Field(default_factory=lambda: f"deal_{uuid.uuid4().hex[:12]}")
    user_id: str
    title: str
    contact_id: Optional[str] = None
    company_id: Optional[str] = None
    value: float = 0
    stage: str = "lead"
    probability: int = 10
    expected_close_date: Optional[datetime] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Task Models
class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    priority: str = "medium"  # low, medium, high
    status: str = "pending"  # pending, in_progress, completed
    related_to: Optional[str] = None  # lead_id, contact_id, deal_id
    related_type: Optional[str] = None  # lead, contact, deal

class Task(BaseModel):
    model_config = ConfigDict(extra="ignore")
    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    user_id: str
    title: str
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    priority: str = "medium"
    status: str = "pending"
    related_to: Optional[str] = None
    related_type: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Activity Models
class ActivityCreate(BaseModel):
    activity_type: str  # email, call, meeting, note
    subject: str
    description: Optional[str] = None
    related_to: Optional[str] = None  # lead_id, contact_id, deal_id
    related_type: Optional[str] = None  # lead, contact, deal

class Activity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    activity_id: str = Field(default_factory=lambda: f"activity_{uuid.uuid4().hex[:12]}")
    user_id: str
    activity_type: str
    subject: str
    description: Optional[str] = None
    related_to: Optional[str] = None
    related_type: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# AI Chat Models
class ChatMessage(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    action_taken: Optional[Dict[str, Any]] = None

# Email Draft Models
class EmailDraftRequest(BaseModel):
    to_name: str
    to_email: str
    purpose: str  # follow_up, proposal, introduction, cold_outreach
    context: Optional[str] = None

class EmailDraftResponse(BaseModel):
    subject: str
    body: str

# ============ AUTHENTICATION HELPER ============

async def get_current_user(request: Request, authorization: Optional[str] = Header(None)) -> User:
    """Get current user from session token (cookie or header)"""
    session_token = None
    
    # Try cookie first
    session_token = request.cookies.get("session_token")
    
    # Fallback to Authorization header
    if not session_token and authorization:
        if authorization.startswith("Bearer "):
            session_token = authorization.replace("Bearer ", "")
    
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Find session
    session_doc = await db.user_sessions.find_one(
        {"session_token": session_token},
        {"_id": 0}
    )
    
    if not session_doc:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    # Check expiry
    expires_at = session_doc["expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Session expired")
    
    # Get user
    user_doc = await db.users.find_one(
        {"user_id": session_doc["user_id"]},
        {"_id": 0}
    )
    
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    
    return User(**user_doc)

# ============ AUTH ENDPOINTS ============

@api_router.post("/auth/session")
async def create_session(response: Response, x_session_id: str = Header(None)):
    """Exchange session_id for user data and set session cookie"""
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header required")
    
    # Get user data from Emergent Auth
    async with httpx.AsyncClient() as client:
        try:
            auth_response = await client.get(
                "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
                headers={"X-Session-ID": x_session_id},
                timeout=10.0
            )
            if auth_response.status_code != 200:
                raise HTTPException(status_code=401, detail="Invalid session ID")
            
            auth_data = auth_response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Auth service error: {str(e)}")
    
    # Check if user exists
    user_doc = await db.users.find_one({"email": auth_data["email"]}, {"_id": 0})
    
    if not user_doc:
        # Create new user
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        user_data = {
            "user_id": user_id,
            "email": auth_data["email"],
            "name": auth_data["name"],
            "picture": auth_data.get("picture"),
            "role": "user",
            "created_at": datetime.now(timezone.utc)
        }
        await db.users.insert_one(user_data)
        user = User(**user_data)
    else:
        user = User(**user_doc)
    
    # Create session
    session_token = auth_data["session_token"]
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    
    session_data = {
        "user_id": user.user_id,
        "session_token": session_token,
        "expires_at": expires_at,
        "created_at": datetime.now(timezone.utc)
    }
    
    await db.user_sessions.insert_one(session_data)
    
    # Set cookie
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="none",
        max_age=7 * 24 * 60 * 60,
        path="/"
    )
    
    return user.model_dump()

@api_router.get("/auth/me")
async def get_me(request: Request, authorization: Optional[str] = Header(None)):
    """Get current user"""
    user = await get_current_user(request, authorization)
    return user.model_dump()

@api_router.post("/auth/logout")
async def logout(request: Request, response: Response, authorization: Optional[str] = Header(None)):
    """Logout user"""
    session_token = request.cookies.get("session_token")
    
    if not session_token and authorization:
        if authorization.startswith("Bearer "):
            session_token = authorization.replace("Bearer ", "")
    
    if session_token:
        await db.user_sessions.delete_one({"session_token": session_token})
    
    response.delete_cookie("session_token", path="/")
    return {"message": "Logged out successfully"}

# ============ LEAD ENDPOINTS ============

@api_router.post("/leads", response_model=Lead)
async def create_lead(lead_data: LeadCreate, request: Request, authorization: Optional[str] = Header(None)):
    """Create a new lead"""
    user = await get_current_user(request, authorization)
    
    lead_dict = lead_data.model_dump()
    lead_obj = Lead(user_id=user.user_id, **lead_dict)
    
    doc = lead_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    doc['updated_at'] = doc['updated_at'].isoformat()
    
    await db.leads.insert_one(doc)
    return lead_obj

@api_router.get("/leads", response_model=List[Lead])
async def get_leads(request: Request, authorization: Optional[str] = Header(None)):
    """Get all leads for current user"""
    user = await get_current_user(request, authorization)
    
    leads = await db.leads.find({"user_id": user.user_id}, {"_id": 0}).to_list(1000)
    
    for lead in leads:
        if isinstance(lead['created_at'], str):
            lead['created_at'] = datetime.fromisoformat(lead['created_at'])
        if isinstance(lead['updated_at'], str):
            lead['updated_at'] = datetime.fromisoformat(lead['updated_at'])
    
    return leads

@api_router.get("/leads/{lead_id}", response_model=Lead)
async def get_lead(lead_id: str, request: Request, authorization: Optional[str] = Header(None)):
    """Get a specific lead"""
    user = await get_current_user(request, authorization)
    
    lead = await db.leads.find_one({"lead_id": lead_id, "user_id": user.user_id}, {"_id": 0})
    
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    if isinstance(lead['created_at'], str):
        lead['created_at'] = datetime.fromisoformat(lead['created_at'])
    if isinstance(lead['updated_at'], str):
        lead['updated_at'] = datetime.fromisoformat(lead['updated_at'])
    
    return lead

@api_router.put("/leads/{lead_id}", response_model=Lead)
async def update_lead(lead_id: str, lead_data: LeadCreate, request: Request, authorization: Optional[str] = Header(None)):
    """Update a lead"""
    user = await get_current_user(request, authorization)
    
    update_data = lead_data.model_dump()
    update_data['updated_at'] = datetime.now(timezone.utc).isoformat()
    
    result = await db.leads.update_one(
        {"lead_id": lead_id, "user_id": user.user_id},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    updated_lead = await db.leads.find_one({"lead_id": lead_id}, {"_id": 0})
    
    if isinstance(updated_lead['created_at'], str):
        updated_lead['created_at'] = datetime.fromisoformat(updated_lead['created_at'])
    if isinstance(updated_lead['updated_at'], str):
        updated_lead['updated_at'] = datetime.fromisoformat(updated_lead['updated_at'])
    
    return updated_lead

@api_router.delete("/leads/{lead_id}")
async def delete_lead(lead_id: str, request: Request, authorization: Optional[str] = Header(None)):
    """Delete a lead"""
    user = await get_current_user(request, authorization)
    
    result = await db.leads.delete_one({"lead_id": lead_id, "user_id": user.user_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    return {"message": "Lead deleted successfully"}

# ============ CONTACT ENDPOINTS ============

@api_router.post("/contacts", response_model=Contact)
async def create_contact(contact_data: ContactCreate, request: Request, authorization: Optional[str] = Header(None)):
    """Create a new contact"""
    user = await get_current_user(request, authorization)
    
    contact_dict = contact_data.model_dump()
    contact_obj = Contact(user_id=user.user_id, **contact_dict)
    
    doc = contact_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    doc['updated_at'] = doc['updated_at'].isoformat()
    
    await db.contacts.insert_one(doc)
    return contact_obj

@api_router.get("/contacts", response_model=List[Contact])
async def get_contacts(request: Request, authorization: Optional[str] = Header(None)):
    """Get all contacts for current user"""
    user = await get_current_user(request, authorization)
    
    contacts = await db.contacts.find({"user_id": user.user_id}, {"_id": 0}).to_list(1000)
    
    for contact in contacts:
        if isinstance(contact['created_at'], str):
            contact['created_at'] = datetime.fromisoformat(contact['created_at'])
        if isinstance(contact['updated_at'], str):
            contact['updated_at'] = datetime.fromisoformat(contact['updated_at'])
    
    return contacts

@api_router.get("/contacts/{contact_id}", response_model=Contact)
async def get_contact(contact_id: str, request: Request, authorization: Optional[str] = Header(None)):
    """Get a specific contact"""
    user = await get_current_user(request, authorization)
    
    contact = await db.contacts.find_one({"contact_id": contact_id, "user_id": user.user_id}, {"_id": 0})
    
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    
    if isinstance(contact['created_at'], str):
        contact['created_at'] = datetime.fromisoformat(contact['created_at'])
    if isinstance(contact['updated_at'], str):
        contact['updated_at'] = datetime.fromisoformat(contact['updated_at'])
    
    return contact

@api_router.put("/contacts/{contact_id}", response_model=Contact)
async def update_contact(contact_id: str, contact_data: ContactCreate, request: Request, authorization: Optional[str] = Header(None)):
    """Update a contact"""
    user = await get_current_user(request, authorization)
    
    update_data = contact_data.model_dump()
    update_data['updated_at'] = datetime.now(timezone.utc).isoformat()
    
    result = await db.contacts.update_one(
        {"contact_id": contact_id, "user_id": user.user_id},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Contact not found")
    
    updated_contact = await db.contacts.find_one({"contact_id": contact_id}, {"_id": 0})
    
    if isinstance(updated_contact['created_at'], str):
        updated_contact['created_at'] = datetime.fromisoformat(updated_contact['created_at'])
    if isinstance(updated_contact['updated_at'], str):
        updated_contact['updated_at'] = datetime.fromisoformat(updated_contact['updated_at'])
    
    return updated_contact

@api_router.delete("/contacts/{contact_id}")
async def delete_contact(contact_id: str, request: Request, authorization: Optional[str] = Header(None)):
    """Delete a contact"""
    user = await get_current_user(request, authorization)
    
    result = await db.contacts.delete_one({"contact_id": contact_id, "user_id": user.user_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Contact not found")
    
    return {"message": "Contact deleted successfully"}

# ============ COMPANY ENDPOINTS ============

@api_router.post("/companies", response_model=Company)
async def create_company(company_data: CompanyCreate, request: Request, authorization: Optional[str] = Header(None)):
    """Create a new company"""
    user = await get_current_user(request, authorization)
    
    company_dict = company_data.model_dump()
    company_obj = Company(user_id=user.user_id, **company_dict)
    
    doc = company_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    doc['updated_at'] = doc['updated_at'].isoformat()
    
    await db.companies.insert_one(doc)
    return company_obj

@api_router.get("/companies", response_model=List[Company])
async def get_companies(request: Request, authorization: Optional[str] = Header(None)):
    """Get all companies for current user"""
    user = await get_current_user(request, authorization)
    
    companies = await db.companies.find({"user_id": user.user_id}, {"_id": 0}).to_list(1000)
    
    for company in companies:
        if isinstance(company['created_at'], str):
            company['created_at'] = datetime.fromisoformat(company['created_at'])
        if isinstance(company['updated_at'], str):
            company['updated_at'] = datetime.fromisoformat(company['updated_at'])
    
    return companies

@api_router.get("/companies/{company_id}", response_model=Company)
async def get_company(company_id: str, request: Request, authorization: Optional[str] = Header(None)):
    """Get a specific company"""
    user = await get_current_user(request, authorization)
    
    company = await db.companies.find_one({"company_id": company_id, "user_id": user.user_id}, {"_id": 0})
    
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    if isinstance(company['created_at'], str):
        company['created_at'] = datetime.fromisoformat(company['created_at'])
    if isinstance(company['updated_at'], str):
        company['updated_at'] = datetime.fromisoformat(company['updated_at'])
    
    return company

@api_router.put("/companies/{company_id}", response_model=Company)
async def update_company(company_id: str, company_data: CompanyCreate, request: Request, authorization: Optional[str] = Header(None)):
    """Update a company"""
    user = await get_current_user(request, authorization)
    
    update_data = company_data.model_dump()
    update_data['updated_at'] = datetime.now(timezone.utc).isoformat()
    
    result = await db.companies.update_one(
        {"company_id": company_id, "user_id": user.user_id},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Company not found")
    
    updated_company = await db.companies.find_one({"company_id": company_id}, {"_id": 0})
    
    if isinstance(updated_company['created_at'], str):
        updated_company['created_at'] = datetime.fromisoformat(updated_company['created_at'])
    if isinstance(updated_company['updated_at'], str):
        updated_company['updated_at'] = datetime.fromisoformat(updated_company['updated_at'])
    
    return updated_company

@api_router.delete("/companies/{company_id}")
async def delete_company(company_id: str, request: Request, authorization: Optional[str] = Header(None)):
    """Delete a company"""
    user = await get_current_user(request, authorization)
    
    result = await db.companies.delete_one({"company_id": company_id, "user_id": user.user_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Company not found")
    
    return {"message": "Company deleted successfully"}

# ============ DEAL ENDPOINTS ============

@api_router.post("/deals", response_model=Deal)
async def create_deal(deal_data: DealCreate, request: Request, authorization: Optional[str] = Header(None)):
    """Create a new deal"""
    user = await get_current_user(request, authorization)
    
    deal_dict = deal_data.model_dump()
    deal_obj = Deal(user_id=user.user_id, **deal_dict)
    
    doc = deal_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    doc['updated_at'] = doc['updated_at'].isoformat()
    if doc['expected_close_date']:
        doc['expected_close_date'] = doc['expected_close_date'].isoformat()
    
    await db.deals.insert_one(doc)
    return deal_obj

@api_router.get("/deals", response_model=List[Deal])
async def get_deals(request: Request, authorization: Optional[str] = Header(None)):
    """Get all deals for current user"""
    user = await get_current_user(request, authorization)
    
    deals = await db.deals.find({"user_id": user.user_id}, {"_id": 0}).to_list(1000)
    
    for deal in deals:
        if isinstance(deal['created_at'], str):
            deal['created_at'] = datetime.fromisoformat(deal['created_at'])
        if isinstance(deal['updated_at'], str):
            deal['updated_at'] = datetime.fromisoformat(deal['updated_at'])
        if deal.get('expected_close_date') and isinstance(deal['expected_close_date'], str):
            deal['expected_close_date'] = datetime.fromisoformat(deal['expected_close_date'])
    
    return deals

@api_router.get("/deals/{deal_id}", response_model=Deal)
async def get_deal(deal_id: str, request: Request, authorization: Optional[str] = Header(None)):
    """Get a specific deal"""
    user = await get_current_user(request, authorization)
    
    deal = await db.deals.find_one({"deal_id": deal_id, "user_id": user.user_id}, {"_id": 0})
    
    if not deal:
        raise HTTPException(status_code=404, detail="Deal not found")
    
    if isinstance(deal['created_at'], str):
        deal['created_at'] = datetime.fromisoformat(deal['created_at'])
    if isinstance(deal['updated_at'], str):
        deal['updated_at'] = datetime.fromisoformat(deal['updated_at'])
    if deal.get('expected_close_date') and isinstance(deal['expected_close_date'], str):
        deal['expected_close_date'] = datetime.fromisoformat(deal['expected_close_date'])
    
    return deal

@api_router.put("/deals/{deal_id}", response_model=Deal)
async def update_deal(deal_id: str, deal_data: DealCreate, request: Request, authorization: Optional[str] = Header(None)):
    """Update a deal"""
    user = await get_current_user(request, authorization)
    
    update_data = deal_data.model_dump()
    update_data['updated_at'] = datetime.now(timezone.utc).isoformat()
    if update_data.get('expected_close_date'):
        update_data['expected_close_date'] = update_data['expected_close_date'].isoformat()
    
    result = await db.deals.update_one(
        {"deal_id": deal_id, "user_id": user.user_id},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Deal not found")
    
    updated_deal = await db.deals.find_one({"deal_id": deal_id}, {"_id": 0})
    
    if isinstance(updated_deal['created_at'], str):
        updated_deal['created_at'] = datetime.fromisoformat(updated_deal['created_at'])
    if isinstance(updated_deal['updated_at'], str):
        updated_deal['updated_at'] = datetime.fromisoformat(updated_deal['updated_at'])
    if updated_deal.get('expected_close_date') and isinstance(updated_deal['expected_close_date'], str):
        updated_deal['expected_close_date'] = datetime.fromisoformat(updated_deal['expected_close_date'])
    
    return updated_deal

@api_router.delete("/deals/{deal_id}")
async def delete_deal(deal_id: str, request: Request, authorization: Optional[str] = Header(None)):
    """Delete a deal"""
    user = await get_current_user(request, authorization)
    
    result = await db.deals.delete_one({"deal_id": deal_id, "user_id": user.user_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Deal not found")
    
    return {"message": "Deal deleted successfully"}

# ============ TASK ENDPOINTS ============

@api_router.post("/tasks", response_model=Task)
async def create_task(task_data: TaskCreate, request: Request, authorization: Optional[str] = Header(None)):
    """Create a new task"""
    user = await get_current_user(request, authorization)
    
    task_dict = task_data.model_dump()
    task_obj = Task(user_id=user.user_id, **task_dict)
    
    doc = task_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    doc['updated_at'] = doc['updated_at'].isoformat()
    if doc['due_date']:
        doc['due_date'] = doc['due_date'].isoformat()
    
    await db.tasks.insert_one(doc)
    return task_obj

@api_router.get("/tasks", response_model=List[Task])
async def get_tasks(request: Request, authorization: Optional[str] = Header(None)):
    """Get all tasks for current user"""
    user = await get_current_user(request, authorization)
    
    tasks = await db.tasks.find({"user_id": user.user_id}, {"_id": 0}).to_list(1000)
    
    for task in tasks:
        if isinstance(task['created_at'], str):
            task['created_at'] = datetime.fromisoformat(task['created_at'])
        if isinstance(task['updated_at'], str):
            task['updated_at'] = datetime.fromisoformat(task['updated_at'])
        if task.get('due_date') and isinstance(task['due_date'], str):
            task['due_date'] = datetime.fromisoformat(task['due_date'])
    
    return tasks

@api_router.put("/tasks/{task_id}", response_model=Task)
async def update_task(task_id: str, task_data: TaskCreate, request: Request, authorization: Optional[str] = Header(None)):
    """Update a task"""
    user = await get_current_user(request, authorization)
    
    update_data = task_data.model_dump()
    update_data['updated_at'] = datetime.now(timezone.utc).isoformat()
    if update_data.get('due_date'):
        update_data['due_date'] = update_data['due_date'].isoformat()
    
    result = await db.tasks.update_one(
        {"task_id": task_id, "user_id": user.user_id},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Task not found")
    
    updated_task = await db.tasks.find_one({"task_id": task_id}, {"_id": 0})
    
    if isinstance(updated_task['created_at'], str):
        updated_task['created_at'] = datetime.fromisoformat(updated_task['created_at'])
    if isinstance(updated_task['updated_at'], str):
        updated_task['updated_at'] = datetime.fromisoformat(updated_task['updated_at'])
    if updated_task.get('due_date') and isinstance(updated_task['due_date'], str):
        updated_task['due_date'] = datetime.fromisoformat(updated_task['due_date'])
    
    return updated_task

@api_router.delete("/tasks/{task_id}")
async def delete_task(task_id: str, request: Request, authorization: Optional[str] = Header(None)):
    """Delete a task"""
    user = await get_current_user(request, authorization)
    
    result = await db.tasks.delete_one({"task_id": task_id, "user_id": user.user_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {"message": "Task deleted successfully"}

# ============ ACTIVITY ENDPOINTS ============

@api_router.post("/activities", response_model=Activity)
async def create_activity(activity_data: ActivityCreate, request: Request, authorization: Optional[str] = Header(None)):
    """Create a new activity"""
    user = await get_current_user(request, authorization)
    
    activity_dict = activity_data.model_dump()
    activity_obj = Activity(user_id=user.user_id, **activity_dict)
    
    doc = activity_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    
    await db.activities.insert_one(doc)
    return activity_obj

@api_router.get("/activities", response_model=List[Activity])
async def get_activities(request: Request, authorization: Optional[str] = Header(None)):
    """Get all activities for current user"""
    user = await get_current_user(request, authorization)
    
    activities = await db.activities.find({"user_id": user.user_id}, {"_id": 0}).sort("created_at", -1).to_list(1000)
    
    for activity in activities:
        if isinstance(activity['created_at'], str):
            activity['created_at'] = datetime.fromisoformat(activity['created_at'])
    
    return activities

# ============ ANALYTICS ENDPOINTS ============

@api_router.get("/analytics/dashboard")
async def get_dashboard_analytics(request: Request, authorization: Optional[str] = Header(None)):
    """Get dashboard analytics"""
    user = await get_current_user(request, authorization)
    
    # Count all entities
    leads_count = await db.leads.count_documents({"user_id": user.user_id})
    contacts_count = await db.contacts.count_documents({"user_id": user.user_id})
    companies_count = await db.companies.count_documents({"user_id": user.user_id})
    deals_count = await db.deals.count_documents({"user_id": user.user_id})
    tasks_count = await db.tasks.count_documents({"user_id": user.user_id, "status": {"$ne": "completed"}})
    
    # Calculate total deal value
    deals = await db.deals.find({"user_id": user.user_id}, {"_id": 0, "value": 1, "stage": 1}).to_list(1000)
    total_value = sum(deal.get('value', 0) for deal in deals)
    won_value = sum(deal.get('value', 0) for deal in deals if deal.get('stage') == 'closed_won')
    
    # Pipeline by stage
    pipeline = {}
    for deal in deals:
        stage = deal.get('stage', 'unknown')
        if stage not in pipeline:
            pipeline[stage] = {"count": 0, "value": 0}
        pipeline[stage]["count"] += 1
        pipeline[stage]["value"] += deal.get('value', 0)
    
    return {
        "leads": leads_count,
        "contacts": contacts_count,
        "companies": companies_count,
        "deals": deals_count,
        "tasks": tasks_count,
        "total_deal_value": total_value,
        "won_deal_value": won_value,
        "pipeline": pipeline
    }

# ============ AI ENDPOINTS ============

@api_router.post("/ai/chat", response_model=ChatResponse)
async def ai_chat(chat_msg: ChatMessage, request: Request, authorization: Optional[str] = Header(None)):
    """AI Sales Copilot - Chat with AI assistant"""
    user = await get_current_user(request, authorization)
    
    # Get user's CRM data for context
    leads_count = await db.leads.count_documents({"user_id": user.user_id})
    contacts_count = await db.contacts.count_documents({"user_id": user.user_id})
    deals_count = await db.deals.count_documents({"user_id": user.user_id})
    
    # Create AI chat with context
    system_message = f"""You are an AI Sales Copilot assistant for a CRM platform. 
The user's current CRM stats: {leads_count} leads, {contacts_count} contacts, {deals_count} deals.
Help them with their sales tasks, answer questions about their data, and provide insights.
Be concise, helpful, and action-oriented."""
    
    try:
        chat = LlmChat(
            api_key=os.environ['EMERGENT_LLM_KEY'],
            session_id=user.user_id,
            system_message=system_message
        ).with_model("openai", "gpt-5.1")
        
        user_message = UserMessage(text=chat_msg.message)
        response = await chat.send_message(user_message)
        
        return ChatResponse(response=response, action_taken=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

@api_router.post("/ai/email-draft", response_model=EmailDraftResponse)
async def ai_email_draft(email_req: EmailDraftRequest, request: Request, authorization: Optional[str] = Header(None)):
    """AI Email Drafting"""
    user = await get_current_user(request, authorization)
    
    purpose_templates = {
        "follow_up": "Write a professional follow-up email",
        "proposal": "Write a business proposal email",
        "introduction": "Write a warm introduction email",
        "cold_outreach": "Write a compelling cold outreach email"
    }
    
    prompt = f"""{purpose_templates.get(email_req.purpose, "Write a professional email")} to {email_req.to_name} ({email_req.to_email}).
Context: {email_req.context or "No additional context provided"}

Generate a professional email with:
1. A compelling subject line
2. A well-structured body

Format your response as:
SUBJECT: [subject line]
BODY: [email body]
"""
    
    try:
        chat = LlmChat(
            api_key=os.environ['EMERGENT_LLM_KEY'],
            session_id=f"{user.user_id}_email",
            system_message="You are an expert email copywriter specializing in business communications."
        ).with_model("openai", "gpt-5.1")
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        
        # Parse response
        lines = response.split('\n')
        subject = ""
        body = ""
        
        for i, line in enumerate(lines):
            if line.startswith("SUBJECT:"):
                subject = line.replace("SUBJECT:", "").strip()
            elif line.startswith("BODY:"):
                body = '\n'.join(lines[i+1:]).strip()
                break
        
        if not subject:
            subject = "Follow up"
        if not body:
            body = response
        
        return EmailDraftResponse(subject=subject, body=body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

@api_router.post("/ai/deal-score/{deal_id}")
async def ai_deal_score(deal_id: str, request: Request, authorization: Optional[str] = Header(None)):
    """AI Deal Scoring - Predict deal probability"""
    user = await get_current_user(request, authorization)
    
    deal = await db.deals.find_one({"deal_id": deal_id, "user_id": user.user_id}, {"_id": 0})
    
    if not deal:
        raise HTTPException(status_code=404, detail="Deal not found")
    
    # Get related activities
    activities_count = await db.activities.count_documents({
        "user_id": user.user_id,
        "related_to": deal_id
    })
    
    # Simple AI-based scoring logic
    prompt = f"""Analyze this deal and predict the probability of closing (0-100%):
Deal Stage: {deal.get('stage')}
Deal Value: ${deal.get('value', 0)}
Activities Count: {activities_count}
Notes: {deal.get('notes', 'No notes')}

Provide a probability score (0-100) and a brief explanation."""
    
    try:
        chat = LlmChat(
            api_key=os.environ['EMERGENT_LLM_KEY'],
            session_id=f"{user.user_id}_scoring",
            system_message="You are an AI sales analyst specializing in deal scoring and predictions."
        ).with_model("openai", "gpt-5.1")
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        
        # Extract score from response (simple heuristic)
        score = 50  # default
        for word in response.split():
            if word.replace('%', '').isdigit():
                score = int(word.replace('%', ''))
                break
        
        # Update deal with new score
        await db.deals.update_one(
            {"deal_id": deal_id},
            {"$set": {"probability": score}}
        )
        
        return {
            "deal_id": deal_id,
            "probability": score,
            "analysis": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

# ============ ROOT ENDPOINT ============

@api_router.get("/")
async def root():
    return {"message": "AI CRM API - Ready to revolutionize sales"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
