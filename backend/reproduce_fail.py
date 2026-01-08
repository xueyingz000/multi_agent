from pydantic import BaseModel
from fastapi import HTTPException
import asyncio

# Mock Session State
session_state = {
    "semantic_results": {
        "alignment_results": {
            "12345": {
                "status": "NEEDS_REVIEW",
                "semantic_category": "BALCONY",
                "confidence": 0.8,
                "reasoning": "Looks like a balcony"
            }
        },
        "hitl_queue": [
            {"guid": "12345", "reason": "Low confidence"}
        ]
    }
}

class ApproveRequest(BaseModel):
    element_guid: str

async def approve_element(req: ApproveRequest):
    """
    Approve the Agent 2 analysis for a specific element.
    """
    guid = req.element_guid
    semantic_data = session_state.get("semantic_results")

    if not semantic_data or guid not in semantic_data.get("alignment_results", {}):
        raise HTTPException(status_code=404, detail="Element analysis not found")

    # 1. Update status in alignment_results
    semantic_data["alignment_results"][guid]["status"] = "VERIFIED"

    # 2. Remove from hitl_queue
    if "hitl_queue" in semantic_data:
        semantic_data["hitl_queue"] = [
            item for item in semantic_data["hitl_queue"] if item["guid"] != guid
        ]

    # Update session state
    session_state["semantic_results"] = semantic_data

    return {"status": "success", "message": "Element approved", "element_id": guid}

async def main():
    try:
        req = ApproveRequest(element_guid="12345")
        res = await approve_element(req)
        print("Success:", res)
        
        # Verify state
        print("State after:", session_state["semantic_results"]["alignment_results"]["12345"]["status"])
        print("Queue length:", len(session_state["semantic_results"]["hitl_queue"]))
        
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
