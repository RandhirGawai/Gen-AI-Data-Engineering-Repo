# AI Voice Agent Architecture - Mortgage Servicing (Complete Interview Q&A + Project Details)

## Architecture at a Glance
Twilio (Phone Call)  
↓ WebSocket (audio stream)  
FastAPI Server (Python)  
↓ PCM16 audio  
Azure AI VoiceLive (GPT-4 Realtime)  
↓ function calls  
Business Logic Services  
↓  
PostgreSQL DB + Servicing Branch API + Twilio API

## Tech Stack

| Layer                  | Technology |
|------------------------|----------|
| Web Framework          | FastAPI + Uvicorn (ASGI) |
| Telephony              | Twilio (calls, WebSocket audio, TwiML) |
| AI Voice Agent         | Azure AI VoiceLive (GPT-4 Realtime) |
| Post-call AI           | Azure OpenAI (GPT-5-nano) — call summaries |
| Database               | PostgreSQL on Azure (psycopg2 + asyncpg) |
| Audio Processing       | audiopop, numpy, soundfile (μ-law → PCM16 conversion) |
| HTTP Client            | httpx (async) |
| Scheduling/Bookings    | Microsoft Graph API (MS Bookings) |
| Deployment             | Azure Web App via GitHub Actions CI/CD |
| Config                 | dotenv environment variables |

## How a Call Works (Step by Step)
1. Call Initiated — System dials customer via Twilio with AMD (Answering Machine Detection)  
2. Twilio Webhook — Server returns TwiML(Twilio Markup Language:Think of TwiML as the "instructions" or "script" for handling voice calls, SMS, MMS, or WhatsApp messages in Twilio applications) that opens a WebSocket audio stream  
3. WebSocket Connected — Server fetches customer data from Servicing Branch API  
4. Agent Initialized — Azure VoiceLive agent loaded with customer context (name, loan, balance), state-driven prompt (GREETING → VERIFY → OCCUPANCY → PAYMENT → WRAPUP), available functions  
5. Audio Streaming — Twilio sends μ-law 8kHz audio → converted to PCM16 24kHz → Azure processes it → response audio sent back  
6. Function Calling — When customer agrees to pay, agent calls payment_now() → hits Servicing Branch payment API  
7. Call End — Transcript sent to Azure OpenAI for summary, call log saved to PostgreSQL

## Key Design Patterns
1. State-Driven Flows — Each call goes through states: GREETING → VERIFY → OCCUPANCY → PAYMENT → WRAPUP. Each state has its own AI prompt.  
2. Multi-tenant Configuration — DB stores flows, guardrails, and agent config per "team" — supports multiple lenders from one codebase.  
3. PII Masking in Logs — Every call gets a CallLogger instance that masks phone numbers, SSNs, account numbers before writing to DB.  
4. Silent State Transitions — Agent calls set_state() internally — customer never hears it.  
5. Audio Format Bridge — Twilio μ-law 8kHz ↔ Azure PCM16 24kHz with real-time conversion.  
6. Thinking Sounds — Keyboard typing sounds play while AI is processing. Audio buffer shared class-level (only 8 bytes per call).  
7. Flow Generator — generator/ folder has Python files that define flows → script generates SQL → inserts into PostgreSQL.

## Specialized Flow Types

| Flow                | Purpose |
|---------------------|--------|
| Standard Collection | Delinquent account calls — verify identity, collect payment |
| TPP/RPP             | Trial Payment Plan / Repayment Plan customers |
| Service Transfer    | Notify customers their loan is being transferred |
| Inbound             | Handle incoming calls, route to right flow |
| Insurance / Sales   | Specialized discovery flows |

## Key APIs Integrated
- Servicing Branch API — Customer lookup, payment processing, memos, next review dates  
- Twilio API — Call creation, AMD, recording, conference transfers  
- Azure AI VoiceLive — Real-time voice AI  
- Azure OpenAI — Post-call summarization  
- Microsoft Graph/Bookings — Schedule appointments

## Current Branch Context
Branch: `bugfix/tpprpp-and-service-ytransfer-greeting-msg-reattempt` is fixing:  
- Agent saying filler phrases before/after function calls — now forbidden  
- IVR detection improvements (silent handling for business phone systems)  
- Call screener handling improvements

## Good Interview Talking Points
- Real-time bidirectional audio over WebSocket with format conversion  
- Function calling architecture — AI decides when to call payment/transfer APIs  
- Business hours enforcement — no transfers outside L2 agent availability  
- AMD (Answering Machine Detection) — async, race-condition handled  
- Multi-tenancy — one system serves multiple mortgage lenders  
- Compliance — PII masking, call logging, NACHA ACH validation  
- Scalability — shared audio buffers, asyncio throughout, Azure cloud deployment

## 1. Real-Time Bidirectional Audio over WebSocket with Format Conversion

**Q1: Explain how you handled real-time bidirectional audio streaming between Twilio and Azure AI.**  
Twilio streams audio over WebSocket using Media Stream protocol as base64-encoded μ-law (G.711) at 8kHz. Azure expects PCM16 at 24kHz. Conversion layer:  
- Incoming: Decode base64 → μ-law bytes → audiopop.ulaw2lin() to PCM16 → audiopop.ratecv() to upsample 8kHz → 24kHz  
- Outgoing: PCM16 24kHz from Azure → audiopop.ratecv() downsample 24kHz → 8kHz → audiopop.lin2ulaw() → base64  
Both directions happen simultaneously in an async loop.

**Q2: Why is μ-law (G.711) used by Twilio instead of raw PCM?**  
μ-law is a companding algorithm used in telephony since the 1960s. It compresses 16-bit PCM into 8-bit using logarithmic scale — more efficient for voice. Twilio uses it because it is the standard codec for PSTN, lower bandwidth (8 bits per sample), optimized for voice intelligibility.

**Q3: What challenges did you face with real-time audio latency?**  
1. Format conversion overhead — minimized by processing chunks in async event loop without blocking.  
2. Sample rate mismatch — 8kHz to 24kHz resampling adds compute.  
3. Azure response latency — during gap, play "thinking sounds" (keyboard typing audio).  
4. WebSocket backpressure — careful buffering; rely on Azure server-side VAD for pacing.

**Q4: How does the WebSocket connection lifecycle work?**  
1. Twilio dials customer + AMD callback registered  
2. Twilio hits /outbound/twiML → Server returns TwiML with <Stream>  
3. Twilio connects to WebSocket URL  
4. Server receives "start" event → extracts call_sid, initializes VoiceLiveAgent  
5. Server receives "media" events → audio chunks → convert → forward to Azure  
6. Azure sends back audio → convert → forward to Twilio as "media" events  
7. Call ends → Twilio sends "stop" event → cleanup, save logs, generate summary

**Q5: What is Server VAD and how does it affect the audio stream?**  
VAD = Voice Activity Detection. Azure’s server handles it. It uses semantic VAD to understand when a sentence is grammatically complete. Prevents interrupting mid-sentence, handles natural pauses. Configured in VoiceLiveAgent initialization with server_vad settings.

## 2. Function Calling Architecture — AI Decides When to Call APIs

**Q6: Explain the function calling architecture in your AI agent.**  
Azure AI VoiceLive supports OpenAI-style function/tool calling. Registered functions: payment_now(), schedule_payment(), transfer_call(), wrapup(), set_state(), callback_schedule(), get_date_info(). When AI decides action, it outputs structured function call JSON. Server intercepts, calls Servicing Branch API, sends result back to AI.

**Q7: How does the AI know WHEN to call a function vs speak?**  
Controlled by state-specific prompts. Example in PAYMENT state: "Once customer confirms routing number, account number, and amount, call payment_now()". Guardrails restrict actions.

**Q8: What happens if a function call fails or the API returns an error?**  
Function dispatch layer handles:  
1. Payment failure — send error to AI as function result; AI tells customer and calls transfer_call().  
2. Network timeout — agent informed and escalates.  
3. Validation failure — local validation (e.g. NACHA), return error immediately to AI.

**Q9: What is the set_state() function and why is it silent?**  
set_state() transitions call to new state internally. Silent because: customer hears nothing, no filler phrases, state changes instantly. Decouples conversation from internal state machine.

**Q10: How do you prevent the AI from hallucinating or making up data?**  
1. Core guardrails injected into every prompt.  
2. Customer data (name, loan, balance) injected at start.  
3. Function result grounding — AI reads back exact API data.  
4. Recent bugfix — check for empty fields before injecting.

## 3. Business Hours Enforcement — No Transfers Outside L2 Availability

**Q11: How do you enforce business hours for agent transfers?**  
Inside transfer_call() handler, call is_within_business_hours() before Twilio transfer. Hours: Mon-Thu 9AM–8PM ET, Fri 9AM–7PM ET, weekends closed. Uses pytz for timezone. Outside hours → return "outside_hours" result; AI offers callback scheduling.

**Q12: What if a customer insists on speaking to someone right now after hours?**  
Agent acknowledges, explains unavailability, offers scheduled callback, logs via callback_schedule(), or ends call with wrapup memo.

**Q13: How do you handle edge cases like holidays or timezone differences?**  
Business hours based on agent availability (Eastern Time). pytz handles DST. Holiday support planned via DB table.

## 5. Multi-Tenancy — One System Serves Multiple Mortgage Lenders

**Q18: How does your system support multiple mortgage lenders?**  
Two-database architecture: shared ai_agent_framework + per-client DB. At call start, fetch LenderID → lookup client DB name → load team config and connect to correct DB.

**Q19: What is the teams table and how does it drive configuration?**  
Central store with team_id, team_name, core_guardrails, functions_available, flows JSONB, modified timestamp. At runtime: inject guardrails, available functions, load state-prompt flows dynamically.

**Q20: How do you ensure one lender’s data doesn’t leak to another?**  
Separate databases, per-call logger with client_id/team_id, API-level isolation, PII masking, no shared state.

## 6. Compliance — PII Masking, Call Logging, NACHA ACH Validation

**Q21: How does PII masking work in your logging system?**  
CallLogger class masks with regex: phone → ***-***-xxxx, SSN → ***-**-xxxx, account/routing numbers masked before any log write.

**Q22: What is NACHA and how do you validate ACH payments?**  
NACHA governs ACH payments. Validate routing number: exactly 9 digits, first two in valid range, checksum (3d1 + 7d2 + ... mod 10 = 0 or 9).

**Q23: What call data do you log and why?**  
Database: call_sid, team_id, client_id, masked numbers, loan_id, duration, status. Application logs (masked): transcript, function calls, state transitions, errors, audio stats. For compliance, debugging, dispute resolution.

**Q24: How do you handle the case where a customer verbally gives a wrong account number?**  
1. AI collects numbers verbally. 2. Reads back for confirmation. 3. Customer confirms. 4. Server validates with NACHA. 5. If valid → API call. 6. If invalid → error to AI, ask repeat. 7. After 2-3 fails → escalate.

## 7. Scalability & Design

**Q25: How did you design the thinking sounds feature for scalability?**  
Class-level shared audio buffer. Each call only tracks 8-byte position. Loaded once, shared across all calls.

**Q26: Why use asyncio throughout instead of threading?**  
I/O-bound workload. Asyncio single event loop handles thousands of connections with low memory vs 1 thread per call.

**Q27: How is the application deployed and what’s the CI/CD pipeline?**  
Azure Web App (App Service). GitHub Actions: push → setup Python → pip install → Azure login → deploy → restart. Secrets in GitHub Secrets + Azure settings.

**Q28: How does your system handle 1,000 concurrent calls?**  
Async event loop, shared buffers, per-call isolation, connection pooling, async HTTP, Azure auto-scaling, stateless design.

**Q29: What would you improve for even better scalability?**  
Redis for shared state, message queue for post-call processing, connection pool tuning, audio optimization, move to AKS.

**Q30: If a call drops mid-payment, how do you handle it?**  
No auto-retry (compliance). Log incomplete attempt. Human follows up. Payment never submitted without full consent.

**Q31: How do you test the agent without making real phone calls?**  
TEST_MODE=True: loads flows from Python files, no live Twilio. Plus ngrok for webhooks, test phone calls, load simulation (50+ concurrent).

**Q32: What security measures are in place?**  
Twilio signature validation, Bearer token auth, Azure Managed Identity, no secrets in code, SSL/TLS everywhere, PII masking, least-privilege DB user.


**Q1: Walk me through exactly how real-time call data is captured during a live call.**  
Every call creates one CallLogger instance in app/utils/call_logger.py. It lives in memory for the entire call.  
It has two in-memory lists:  
- self.logs: List[Dict] — every event (errors, state changes, function calls, audio events)  
- self.transcript: List[Dict] — only spoken turns (customer + agent)  

As the call progresses, different parts call methods on this logger:

| What Happens                  | Method Called                        | Stored In                  |
|-------------------------------|--------------------------------------|----------------------------|
| Customer speaks               | log_user_speech(text)                | transcript + logs          |
| Agent speaks                  | log_agent_speech(text)               | transcript + logs          |
| Agent calls a function        | log_function_call(name, args, result)| logs + DB immediately      |
| State changes                 | log_state_change(field, old, new)    | logs                       |
| Identity verification attempt | log_verification_step(type, status)  | logs + DB immediately      |
| Birthday/nicety message       | log_nicety(type, message)            | logs + DB immediately      |
| Who agent spoke with          | log_spoke_with(person_name)          | logs + DB immediately      |

Some events go to DB immediately; others wait until end of call.

**Q2: How does the transcript get built in real-time?**  
Each time Azure VoiceLive finalizes a speech segment, azure_voice.py handler calls log_user_speech() or log_agent_speech().  
Each turn stored as:  
{"role": "user/agent", "text": "...", "timestamp": "...", "elapsed_ms": ..., "is_final": True}  
At end, get_transcript_text() formats it as timestamped lines (e.g. [00:00:12] Agent: Hello...). This text is stored in PostgreSQL and sent to Azure OpenAI.

**Q3: What is the CallLoggerRegistry and why does it exist?**  
Class holding Dict[str, CallLogger] for all concurrent calls.  
Supports lookup by call_id (internal UUID), call_sid (Twilio), or stream_sid (WebSocket).  
get() searches any of the three IDs. Has cleanup_old() to remove loggers older than 1 hour (prevents memory leaks).

**Q4: How does PII masking happen in real-time before data is captured?**  
CallLogger has PII_FIELDS list: phone, dob, ssn, account_number, loan_number, etc.  
Every log method with extra dict passes through mask_pii_in_dict() (recursive).  
mask_value() keeps last 4 characters, replaces rest with ***. Masking is automatic and cannot be bypassed.

**Q5: How does the call data get written to the database? Explain the full lifecycle.**  
Three phases:  
1. CREATE (call starts): call_logger.create_db_record() → INSERT into call_logs with status="In Progress"  
2. UPDATE (during call): call_logger.update_db_record() → dynamic SQL UPDATE for changed fields  
3. FINALIZE (call ends): call_logger.finalize_db_record() → UPDATE with final status, duration, transcript, AI summary, quality, intent, end time

**Q6: How does the dynamic SQL UPDATE work?**  
update_call_log() builds SQL only with non-None fields:  
UPDATE call_logs SET call_status = %s, call_duration_sec = %s ... WHERE call_id = %s  
Minimal writes, no wasted NULL overwrites.

**Q7: How are call_activity writes done without blocking the real-time audio stream?**  
When logging activity: asyncio.create_task(self._log_activity_to_db(...))  
Wrapped with run_in_executor() because psycopg2 is sync. Audio loop continues immediately; DB write runs in thread pool.

**Q8: How does the call_activity DB write guard against writing before client_id is available?**  
Guard in _log_activity_to_db(): if client_id is None → skip (log only to memory). Once customer data fetched, writes proceed.

**Call Activity Flow Summary**  
CALL STARTS → CallLogger created → INSERT "In Progress"  
DURING CALL → speech to transcript[], function calls → async task → run_in_executor → psycopg2 INSERT  
CALL ENDS → format transcript + generate summary → UPDATE with final data

## Post-Call AI Summary

**Q10: How does the AI call summary get generated and stored?**  
finalize_db_record() triggers generate_call_summary():  
1. Format transcript into speaker:text lines  
2. Call Azure OpenAI (gpt-5-nano, temperature=0.3) with summarizer prompt  
3. Store result in call_logs.call_summary column (2-4 factual sentences)

## Database Connection Management

**Q11: How is the database connection managed?**  
get_db_connection() creates fresh psycopg2 connection per operation (autocommit=True). Supports shared ai_agent_framework DB and client-specific DB.

**Q12: Why use psycopg2 (sync) instead of asyncpg?**  
Thread safety in run_in_executor(). psycopg2 works cleanly in thread pool; asyncpg is not thread-safe.

Current branch: bugfix/tpprpp-and-service-ytransfer-greeting-msg-reattempt (fixing filler phrases, IVR detection, call screener).

