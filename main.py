from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
import json
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
import os
from motor.motor_asyncio import AsyncIOMotorClient

# MongoDB connection setup (using Motor for async support)
MONGODB_URI = os.getenv("MONGODB_URI")  # Use environment variable for MongoDB URI
if MONGODB_URI is None:
    raise ValueError("MONGODB_URI environment variable is not set")

client = AsyncIOMotorClient(MONGODB_URI)
db = client['hosting_company']
customers_collection = db['customers']
plans_collection = db['plans']

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

# Pydantic Model to validate incoming customer data
class CustomerCreate(BaseModel):
    name: str = Field(..., description="Name of the customer")
    subscribed_plan: str = Field(..., description="Name of the plan the customer is subscribed to")
    renewal_date: datetime = Field(..., description="Renewal date for the subscription")
    average_usage: int = Field(..., description="Average usage by the customer", ge=0)

# Define the structure of the result
class SupportResult(BaseModel):
    support_advice: str = Field(description="Advice returned to the customer")
    block_card: bool = Field(description="Whether to block the customer's card")
    risk: int = Field(description="Risk level of query", ge=0, le=10)
    escalation_summary: str = Field(description="Summary for L2 escalation, if required", default="")

# Define dependencies
@dataclass
class SupportDependencies:
    customer_id: ObjectId
    db: AsyncIOMotorClient

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

# Customer Creation Endpoint
@app.post("/customers/create", response_model=CustomerCreate)
async def create_customer(customer: CustomerCreate):
    new_customer = customer.dict()
    result = await customers_collection.insert_one(new_customer)
    customer_data = await customers_collection.find_one({"_id": result.inserted_id})
    return customer_data

@support_agent.tool
async def customer_plan(ctx: RunContext[SupportDependencies]) -> dict:
    customer_id = ctx.deps.customer_id
    print(f"customer_id before any conversion: {customer_id}, Type: {type(customer_id)}")

    if isinstance(customer_id, str):
        try:
            customer_id = ObjectId(customer_id)
        except Exception as e:
            print(f"Error converting to ObjectId: {e}") # Log conversion errors!
            return {"error": f"Invalid ObjectId format: {e}"}

    print(f"customer_id AFTER conversion attempt: {customer_id}, Type: {type(customer_id)}") # CRUCIAL CHECK
    print(f"Querying customer with ID: {customer_id}")

    try:
        customer = await ctx.deps.db.customers_collection.find_one({"_id": customer_id})
        print(f"Customer found: {customer}")  # Log the retrieved customer data
        if customer is None:
            print("No customer found with the provided ID.")  # Log if no customer is found
            return {
                "support_advice": "I was unable to find your account. Please check your customer ID and try again.",
                "block_card": False,
                "risk": 0,
                "escalation_summary": ""
            }
        else:
            # Check if the customer's plan is active
            if customer['subscribed_plan'] == "Basic Hosting" and customer['average_usage'] > 70:
                return {
                    "support_advice": "Your plan is active, but you are exceeding your average usage. Consider upgrading your plan.",
                    "block_card": False,
                    "risk": 5,
                    "escalation_summary": ""
                }
            elif customer['renewal_date'] < datetime.now():
                return {
                    "support_advice": "Your renewal date has passed. Please renew your plan to avoid service interruption.",
                    "block_card": False,
                    "risk": 9,
                    "escalation_summary": "Customer's renewal date is past due."
                }
            else:
                return {
                    "support_advice": "Your plan is active. If you have any issues, please contact support.",
                    "block_card": False,
                    "risk": 0,
                    "escalation_summary": ""
                }
    except Exception as e:
        print(f"Database query error: {e}")
        return {"error": f"Database query error: {e}"}

@support_agent.tool
async def list_available_plans(ctx: RunContext[SupportDependencies]) -> List[dict]:
    plans = await ctx.deps.db.plans_collection.find().to_list(length=10)
    return [{"name": plan["name"], "description": plan["description"], "cost": plan["cost"]} for plan in plans]

@support_agent.tool
async def escalate_to_l2(ctx: RunContext[SupportDependencies], issue_summary: str) -> str:
    customer = await ctx.deps.db.customers_collection.find_one({"_id": ObjectId(ctx.deps.customer_id)})
    if customer:
        await ctx.deps.db.customers_collection.update_one(
            {"_id": ObjectId(ctx.deps.customer_id)},
            {"$set": {"escalation_log": issue_summary}}
        )
        return (
            f"Escalation Summary:\nCustomer: {customer['name']}\nPlan: {customer['subscribed_plan']}\n"
            f"Issue: {issue_summary}"
        )
    return "Escalation failed: Customer not found."

def str_to_objectid(v: str) -> ObjectId:
    try:
        return ObjectId(v)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ObjectId format: {e}")

class SupportQuery(BaseModel):
    customer_id: str = Field(..., alias="customer_id", description="MongoDB ObjectId as string")
    query: str

    @classmethod
    def parse_obj(cls, obj: dict) -> "SupportQuery":
        try:
            obj["customer_id"] = str_to_objectid(obj["customer_id"])
            return super().parse_obj(obj)
        except HTTPException as e:
            raise e  # Re-raise the HTTPException
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

@app.post("/support")
async def support_query(payload: SupportQuery):
    try:
        print(f"Received payload: {payload}")  # Log the received payload
        deps = SupportDependencies(customer_id=payload.customer_id, db=db)
        print(f"Running agent with customer ID: {payload.customer_id} and query: {payload.query}")  # Log the customer ID and query
        result = await support_agent.run(payload.query, deps=deps)

        print(f"Response from agent: {result}")

        if hasattr(result, 'data') and isinstance(result.data, SupportResult):
            result_data = result.data
            return {
                "support_advice": result_data.support_advice,
                "block_card": result_data.block_card,
                "risk": result_data.risk,
                "escalation_summary": result_data.escalation_summary
            }
        else:
            raise HTTPException(status_code=500, detail=f"Unexpected response from agent: {result}")
    except Exception as e:
        print(f"Error in support_query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing support query: {e}")
    
@app.get("/test_lookup/{customer_id}")
async def test_lookup(customer_id: str):
    try:
        obj_id = ObjectId(customer_id)
        customer = await customers_collection.find_one({"_id": obj_id})
        return customer
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

