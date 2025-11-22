"""
B2B Lead Qualification Voice Agent
Qualifies leads through a conversational interface by asking a series of questions.
"""

from livekit.plugins import aws, deepgram, silero, langchain
from livekit import agents, api
from livekit.protocol import sip as sip_protocol
from livekit.agents import AgentSession, Agent, RunContext, get_job_context

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import sys

from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from logging import getLogger
import asyncio

# Initialize logger
logger = getLogger(__name__)
logger.setLevel("INFO")

# Load environment variables from .env file
load_dotenv("./.env")

class Settings(BaseSettings):

    DEEPGRAM_API_KEY: str

    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_DEFAULT_REGION: str

    GEMINI_API_KEY: str
    GEMINI_MODEL_NAME: str 

    TWILIO_SIP_DOMAIN: str
    TWILIO_PHONE_NUMBER: str
    TWILIO_SIP_USERNAME: str
    TWILIO_SIP_PASSWORD: str

    LIVEKIT_API_KEY: str
    LIVEKIT_API_SECRET: str
    LIVEKIT_URL: str

settings = Settings()

# LLM for livekit voice agent
class ChatModel(ChatGoogleGenerativeAI):
    def __init__(
            self,
            api_key: str,
            model: str = "gemini-2.0-flash",
            temperature: float = 0.2
            ):
        
        super().__init__(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )

# CRM Integration Placeholder( Implement based on specific CRM APIs)
async def update_crm(lead_info: dict):
    # Placeholder function to simulate CRM update
    logger.info(f"Updating CRM with lead info: {lead_info}")
    # Implement actual CRM API calls here

# --- 1. LangGraph & Model Setup ---

class AgentState(MessagesState):
    pass

async def model_node(state: AgentState):
    """Single node that runs the model."""
    llm = ChatModel(
        api_key=settings.GEMINI_API_KEY,
        model=settings.GEMINI_MODEL_NAME,
    )
    
    messages = state.get("messages", [])
    
    # Guard: Gemini requires at least one non-system message (HumanMessage/AIMessage).
    # If only system messages or no content is present, prepend a minimal HumanMessage.
    has_non_system = False
    for m in messages:
        if isinstance(m, (HumanMessage, AIMessage)):
            has_non_system = True
            break
        if isinstance(m, dict):
            role = (m.get("role") or m.get("type") or "").lower()
            content = m.get("content") or m.get("text") or None
            if content and role != "system":
                has_non_system = True
                break
    
    if not has_non_system:
        # Provide a safe, minimal human message so the Gemini API accepts the request.
        messages = [HumanMessage(content="Remember You are an AI Sales Development Representative.")] + list(messages)
    
    response = llm.invoke(messages)
    
    return {"messages": [response]}

def create_graph():
    """Builds and compiles the LangGraph."""
    builder = StateGraph(AgentState)
    builder.add_node("agent", model_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile()


class QualificationAgent(Agent):
    """AI SDR that qualifies B2B leads using LangGraph"""
    
    def __init__(self, graph_runnable):

        langchain_llm = langchain.LLMAdapter(graph_runnable)
        self.ctx = ctx

        super().__init__(
            llm=langchain_llm,
            instructions="""
                You are Voice Call Agent, a friendly and efficient Sales Development Representative at XYZ Company. 
                You are calling a lead who recently inquired about your B2B SaaS platform.

                **OBJECTIVE:**
                Your goal is to qualify the lead using the BANT framework (Budget, Authority, Need, Timeline) and schedule a follow-up meeting if they are a good fit.

                **CONVERSATION STYLE (CRITICAL):**
                - **Be Concise:** Speak like a human on the phone. Keep responses under 2 short sentences. No paragraphs.
                - **One Question Rule:** Never ask two questions in a row. Ask one, wait for the answer.
                - **Listen:** If the user interrupts, stop talking immediately.

                **QUALIFICATION FRAMEWORK (BANT):**
                Do not ask these like a checklist. Weave them into the conversation naturally.
                1. **Need:** "To start, what specific challenges are you hoping our platform can help solve?"
                2. **Authority:** "Who else on your team would be weighing in on this decision?"
                3. **Timeline:** "Ideally, when are you hoping to have a solution in place?"
                4. **Budget:** "Do you have a specific budget range allocated for this project?"

                **OBJECTION HANDLING:**
                - If they say **"I'm busy"**: "I completely understand. I can be very brief—just 60 seconds to see if this is even relevant for you. Is that okay?"
                """,
        )

async def entrypoint(ctx: agents.JobContext):
    """
    Entry point for the LiveKit Voice Agent job.
    """

    # Initialize Deepgram ASR
    stt = deepgram.STT(
        api_key=settings.DEEPGRAM_API_KEY
    )

    # Initialize AWS Polly TTS
    tts = aws.TTS(
        voice="Joanna",
        region=settings.AWS_DEFAULT_REGION,
        api_key=settings.AWS_ACCESS_KEY_ID,
        api_secret=settings.AWS_SECRET_ACCESS_KEY
    )

    # Initialize Silero VAD
    vad = silero.VAD.load()

    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    graph = create_graph()

    session = AgentSession(
        stt=stt,
        tts=tts,
        vad=vad,
    )

    agent = QualificationAgent(graph_runnable=graph)

    await session.start(room=ctx.room, agent=agent)

    greeting = """
    Say: 'Hi! This is Solayappan from XYZ company. You recently inquired about our platform. Do you have 2 minutes to discuss your needs?
    """

    await session.say(text=greeting)

    logger.info("\nQualification call started.\n")

async def create_outbound_sip_trunk_if_not_exists() -> str:
    
    livekit_api = api.LiveKitAPI(
        api_key=settings.LIVEKIT_API_KEY,
        api_secret=settings.LIVEKIT_API_SECRET,
        url=settings.LIVEKIT_URL,
    )

    try:
        # List existing trunks
        list_request = sip_protocol.ListSIPOutboundTrunkRequest()
        existing_trunks = await livekit_api.sip.list_sip_outbound_trunk(list=list_request)
        
        # Check if trunk already exists
        for trunk in existing_trunks.items:
            if trunk.name == "LoanRepaymentTrunk":
                print("SIP Trunk already exists.")
                print(f"Using existing trunk: {trunk.sip_trunk_id}")
                return trunk.sip_trunk_id
            
        logger.info("\nCreating new SIP Outbound Trunk...\n")
        new_trunk_info = sip_protocol.SIPOutboundTrunkInfo(
            name="LoanRepaymentTrunk",
            address=settings.TWILIO_SIP_DOMAIN,
            numbers=[settings.TWILIO_PHONE_NUMBER],
            auth_password=settings.TWILIO_SIP_PASSWORD,
            auth_username=settings.TWILIO_SIP_USERNAME,
        )

        create = sip_protocol.CreateSIPOutboundTrunkRequest(
            trunk=new_trunk_info
        )

        trunk = await livekit_api.sip.create_sip_outbound_trunk(create)
        logger.info(f"\nCreated new SIP Trunk: {trunk.sip_trunk_id}\n")
        new_trunk_id = trunk.sip_trunk_id

        return new_trunk_id
    finally:
        await livekit_api.aclose()

async def make_call(phone: str):
    """Initiate Outbound call to lead using Twilio SIP"""

    livekit_api = api.LiveKitAPI(
        api_key=settings.LIVEKIT_API_KEY,
        api_secret=settings.LIVEKIT_API_SECRET,
        url=settings.LIVEKIT_URL
    )

    trunk_id = await create_outbound_sip_trunk_if_not_exists()

    try:

        create = sip_protocol.CreateSIPParticipantRequest(
            sip_trunk_id=trunk_id,
            sip_call_to=phone,
            room_name=f"qualify : {phone}",
            participant_identity=phone,
            participant_name="Lead",
            krisp_enabled=True,
            wait_until_answered=True,
        )

        logger.info(f"\nCalling {phone}...\n")

        await livekit_api.sip.create_sip_participant(
            create=create
        )

        logger.info(f"\nCall initiated to {phone}\n")

    except Exception as e:
        logger.error(f"\nCall failed: {e}\n")
    finally:
        await livekit_api.aclose()

if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1].startswith('+'):
        phone_number = sys.argv[1]
        asyncio.run(make_call(phone_number))
    elif len(sys.argv) > 1 and sys.argv[1] == "agent":
        sys.argv.pop(1)
        agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
    else:
        print("Usage:")
        print("  uv run main.py agent dev            # to run the agent worker")
        print("  uv run main.py +1234567890 dev     # to make an outbound call to the specified phone number in E.164 format")
    