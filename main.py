from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from pydantic_ai import Agent, RunContext
from typing import List
from datetime import datetime
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# MongoDB connection setup
MONGODB_URI = os.getenv("MONGODB_URI")
if MONGODB_URI is None:
    raise ValueError("MONGODB_URI environment variable is not set")

client = AsyncIOMotorClient(MONGODB_URI)
db = client["hosting_company"]
customers_collection = db["customers"]
plans_collection = db["plans"]

# FastAPI app setup
app = FastAPI()

@app.on_event("startup")
async def startup_db():
    try:
        await db.command("ping")
        print("MongoDB connection successful")
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        raise HTTPException(status_code=500, detail="MongoDB connection failed")

# Define dependencies
@dataclass
class SupportDependencies:
    customer_id: ObjectId
    db: AsyncIOMotorClient

# Define the structure of the result
class SupportResult(BaseModel):
    support_advice: str = Field(description="Advice returned to the customer")
    block_card: bool = Field(description="Whether to block the customer's card")
    risk: int = Field(description="Risk level of query", ge=0, le=10)
    escalation_summary: str = Field(description="Summary for L2 escalation, if required", default="")

# Create the support agent
support_agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=SupportDependencies,
    result_type=SupportResult,
    system_prompt=(
        f"""You are a Level 1 support agent for a hosting and data warehousing company. Handle queries related to the customer's subscribed plan, provide suggestions for advanced features, and escalate high-risk issues to Level 2 with a detailed summary.
        For your reference :
        Current Date and Time is : {datetime.now()}
        Following are some general rules regarding:
        1. If the customer's website is down, check if their plan is active and if they have exceeded their average usage.
        2. If the renewal date is approaching, advise the customer to renew their plan.
        3. Users with more than 70% average resource consumption should be advised to upgrade to a higher plan.
        4. If there is a downtime despite the plan being active, involve higher-level support from engineers.
        """
    ),
)

# Tool to fetch customer plan details
@support_agent.tool
async def customer_plan(ctx: RunContext[SupportDependencies]) -> dict:
    """Fetches the customer's current subscribed plan and its details."""
    try:
        customer = await ctx.deps.db.customers_collection.find_one({"_id": ctx.deps.customer_id})
        if customer:
            return {
                "plan": customer["subscribed_plan"],
                "renewal_date": customer["renewal_date"],
                "average_usage": customer["average_usage"],
            }
        return {"error": "Customer not found"}
    except Exception as e:
        return {"error": f"Database query error: {e}"}

# Tool to list available plans
@support_agent.tool
async def list_available_plans(ctx: RunContext[SupportDependencies]) -> List[dict]:
    """Returns a list of available plans and their descriptions."""
    print("hello")
    plans = await ctx.deps.db.plans_collection.find().to_list(length=10)
    print(plans)
    return [{"name": plan["name"], "description": plan["description"], "cost": plan["cost"]} for plan in plans]

# Tool to escalate issues to Level 2
@support_agent.tool
async def escalate_to_l2(ctx: RunContext[SupportDependencies], issue_summary: str) -> str:
    """Escalates the issue to Level 2 support with a summary."""
    try:
        customer = await ctx.deps.db.customers_collection.find_one({"_id": ctx.deps.customer_id})
        if customer:
            await ctx.deps.db.customers_collection.update_one(
                {"_id": ctx.deps.customer_id},
                {"$set": {"escalation_log": issue_summary}}
            )
            return (
                f"Escalation Summary:\nCustomer: {customer['name']}\nPlan: {customer['subscribed_plan']}\n"
                f"Issue: {issue_summary}"
            )
        return "Escalation failed: Customer not found."
    except Exception as e:
        return f"Escalation error: {e}"

# Helper function to validate and convert customer ID
def str_to_objectid(customer_id: str) -> ObjectId:
    try:
        return ObjectId(customer_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid customer ID format: {e}")

# Pydantic model for support query
class SupportQuery(BaseModel):
    customer_id: str = Field(..., description="MongoDB ObjectId as string")
    query: str

    @classmethod
    def validate_customer_id(cls, customer_id: str) -> ObjectId:
        return str_to_objectid(customer_id)

@app.post("/support")
async def support_query(payload: SupportQuery):
    try:
        customer_id = str_to_objectid(payload.customer_id)
        deps = SupportDependencies(customer_id=customer_id, db=db)
        result = await support_agent.run(payload.query, deps=deps)

        if hasattr(result, "data") and isinstance(result.data, SupportResult):
            return {
                "support_advice": result.data.support_advice,
                "block_card": result.data.block_card,
                "risk": result.data.risk,
                "escalation_summary": result.data.escalation_summary,
            }
        else:
            raise HTTPException(status_code=500, detail=f"Unexpected response from agent: {result}")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing support query: {e}")

from fastapi.encoders import jsonable_encoder
from bson import ObjectId
from typing import List

# Helper function to convert ObjectId to string
def convert_objectid(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, dict):
        return {key: convert_objectid(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_objectid(item) for item in obj]
    return obj

@app.get("/plans")
async def get_plans():
    try:
        plans = await plans_collection.find().to_list(length=10)
        # Convert any ObjectId fields to string
        plans = convert_objectid(plans)
        return jsonable_encoder(plans)  # Now return the jsonable_encoder version
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching plans: {e}")



from fastapi.encoders import jsonable_encoder
from bson import ObjectId
from typing import Dict

# Custom encoder to handle ObjectId conversion
# Convert ObjectId to string before returning
def convert_objectid_to_str(obj: Dict):
    """
    Convert ObjectId fields to string recursively in the dictionary.
    """
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_objectid_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid_to_str(item) for item in obj]
    return obj

@app.get("/customer/{customer_id}")
async def get_customer(customer_id: str):
    try:
        # Convert customer_id from string to ObjectId
        obj_id = str_to_objectid(customer_id)
        
        # Fetch customer from the database
        customer = await customers_collection.find_one({"_id": obj_id})
        
        if customer:
            # Convert the customer data to JSON-safe format (convert ObjectId to string)
            customer = convert_objectid_to_str(customer)
            return customer
        else:
            raise HTTPException(status_code=404, detail="Customer not found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching customer: {e}")




if __name__ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
