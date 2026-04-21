# AWS Bedrock Interview Questions & Answers - Complete Guide

**Complete Research-Based Interview Preparation Document**  
*Compiled from blogs, AWS documentation, GitHub, Reddit, and technical forums*

---

## Table of Contents

1. [Fundamental Concepts](#fundamental-concepts)
2. [Models and Model Selection](#models-and-model-selection)
3. [RAG (Retrieval Augmented Generation)](#rag-retrieval-augmented-generation)
4. [Agents](#agents)
5. [Fine-tuning and Customization](#fine-tuning-and-customization)
6. [Cost Optimization](#cost-optimization)
7. [Security and Guardrails](#security-and-guardrails)
8. [Scenario-Based Questions](#scenario-based-questions)
9. [Advanced Topics](#advanced-topics)
10. [Summary Comparison Table](#summary-comparison-table)

---

## FUNDAMENTAL CONCEPTS

### Q: What is AWS Bedrock and how does it differ from other AWS AI services?

**A:** AWS Bedrock is a fully managed cloud service that gives you easy access to powerful AI models created by companies like Anthropic, Meta, Stability AI, and others. Unlike services like Amazon Rekognition (which has specific AI features built-in) or SageMaker (where you manage training yourself), Bedrock acts like a bazaar—a big marketplace where different AI models are available under one roof. You don't manage any servers or infrastructure. You just pick a model, send your data to it, and get results back. Bedrock is model-agnostic, meaning it supports many different models, not just Amazon's own models.

---

### Q: What are Foundation Models and what role do they play in AWS Bedrock?

**A:** Foundation Models are very large, general-purpose AI systems trained on huge amounts of text and image data from the internet. They are like master students who have learned from billions of examples. Because of this broad learning, they can do many different tasks without special training—write text, answer questions, generate images, translate languages, and more. In AWS Bedrock, Foundation Models are provided by companies like Anthropic, Meta, Cohere, AI21 Labs, and Stability AI. These models are already pre-trained (ready to use), so you don't need to train them from scratch.

---

### Q: How does AWS Bedrock maintain data privacy?

**A:** AWS Bedrock is designed to be a "privacy vault" for your AI work. Your data never leaves your AWS account and is never used to train AWS's own models. This means your sensitive information stays private. All data is encrypted when it travels to Bedrock (encryption in transit) and when it's stored (encryption at rest). You control who can access your data through AWS IAM (Identity and Access Management) policies. This is especially important for businesses in healthcare, finance, and other regulated industries that must keep data private.

---

### Q: What are the main advantages of using AWS Bedrock?

**A:** The main benefits are:

1. **No infrastructure management** — you don't run servers
2. **Access to many models** from different companies in one place
3. **Fast deployment** — you can start using AI in minutes, not months
4. **Pay only for what you use** — no upfront costs or licensing fees
5. **Built-in safety features** through Guardrails
6. **Seamless integration** with other AWS services like S3, Lambda, and DynamoDB
7. **Existing AWS security** and compliance setup

---

## MODELS AND MODEL SELECTION

### Q: Name the main Foundation Models available in AWS Bedrock and their use cases.

**A:** Here are the common ones:

- **Claude (Anthropic)** — best for writing, reasoning, and having smart conversations
- **Llama 2 (Meta)** — good for many tasks, open-source, cost-effective
- **Jurassic (AI21 Labs)** — strong for text generation and Q&A
- **Titan (Amazon)** — good for embeddings and text tasks
- **Stable Diffusion (Stability AI)** — for generating images from text descriptions

You choose based on your task — if you need writing and reasoning, use Claude; if you need image generation, use Stable Diffusion.

---

### Q: How do you select the right Foundation Model for your use case?

**A:** Start by thinking about your task. Ask:

- What do I need — text generation, classification, image generation, or embeddings?
- How fast must it respond?
- What's my budget?

Test multiple models using Bedrock's playground (a free testing area in the console) with real examples from your work. Compare them on three things:

1. **Quality** — how good are the answers?
2. **Speed** — how fast does it respond?
3. **Cost** — how much do tokens cost?

For simple tasks, use smaller, cheaper models like Claude 3 Haiku. For complex tasks, use larger models like Claude 3 Sonnet or Opus. AWS provides evaluation tools to help you benchmark models.

---

### Q: What is the difference between Claude Haiku, Sonnet, and Opus models?

**A:** These are three versions of Anthropic's Claude model, each with different capabilities:

- **Claude Haiku** — smallest and fastest; good for quick tasks, light chatbots, and when cost matters most
- **Claude Sonnet** — in the middle; balances quality and speed well, works for most tasks like customer support and content creation
- **Claude Opus** — largest and smartest; best for complex reasoning, analysis, and creative writing, but slower and more expensive

Think of it like small, medium, and large coffee sizes — smaller is faster and cheaper, larger is richer and more powerful.

---

## RAG (RETRIEVAL AUGMENTED GENERATION)

### Q: What is Retrieval Augmented Generation (RAG) and why is it important?

**A:** RAG is a technique that combines AI models with your own data. Normally, AI models only know what they learned during training (like a student who stopped learning years ago). RAG lets you teach the model about your specific information — company documents, policies, customer data, anything you have.

It works like this:

1. User asks a question
2. System searches your documents for relevant info
3. System gives that info to the AI model
4. Model uses the info to answer the question accurately

RAG is crucial because it fixes the biggest problem with AI — hallucinations (making up wrong information). With RAG, answers are grounded in real company data.

---

### Q: How does Amazon Bedrock Knowledge Bases work in a RAG system?

**A:** Knowledge Bases is AWS's fully managed RAG service. You upload your documents (PDFs, Word docs, plain text) to an S3 bucket. The Knowledge Base automatically:

1. Breaks documents into small chunks
2. Converts them into embeddings (mathematical representations)
3. Stores them in a vector database

When a user asks a question:

1. System converts the question into embeddings
2. Finds the most similar document chunks using vector search
3. Sends those chunks plus the question to the AI model
4. The model generates an answer based on the retrieved documents

You don't need to manage vector databases or write complex code — Bedrock handles all the technical work.

---

### Q: What is a vector embedding and why is it used in RAG?

**A:** A vector embedding is a list of numbers (like coordinates in space) that represents the meaning of text. For example, the word "king" and the phrase "male ruler" would have similar vector embeddings because they mean similar things.

In RAG, embeddings help the system quickly find relevant documents without reading every word. It's like having a fingerprint for meaning — documents with similar meanings have similar embeddings.

When you ask a question, the system converts your question into embeddings and finds document chunks with similar embeddings. This is much faster than searching word-by-word and works even when people use different words with the same meaning.

---

### Q: Describe a practical scenario where RAG is essential.

**A:** **Scenario: Bank Customer Support System**

- **Old way**: hire people to read the loan handbook and answer questions
- **New way with RAG**:
  1. Upload all loan documents, policies, FAQs to Knowledge Base
  2. Customer asks "What is my interest rate adjustment schedule?"
  3. RAG finds the relevant policy documents
  4. AI model reads the documents and gives a specific, accurate answer
  5. For unusual questions, the system can escalate to a human

**Benefits**:
- Customers get instant, accurate answers
- No hallucinations
- Consistent answers
- Saves money on staff
- Works 24/7

---

### Q: What happens when RAG retrieves irrelevant documents?

**A:** When the wrong documents are retrieved, the model gets bad information and gives a bad answer. For example, if a user asks about "pricing" and the system retrieves documents about "pricing strategy" instead of "product pricing," the answer will be wrong.

To fix this:

1. **Use better chunking** (breaking documents into better pieces)
2. **Add metadata tags** to documents so the system can filter by category
3. **Use reranking models** that re-score retrieved documents for relevance
4. **Fine-tune embeddings** on your specific data
5. **Implement hybrid search** (combine semantic search with keyword search)
6. **Regularly test and monitor** RAG quality

Amazon Bedrock Knowledge Bases now supports advanced chunking (semantic, hierarchical) and reranking models to solve this problem.

---

## AGENTS

### Q: What are AWS Bedrock Agents and what problems do they solve?

**A:** Agents are AI systems that can think, plan, and take action on your behalf. While a simple chatbot just answers questions, an agent can:

- Break down complex tasks into steps
- Decide which tools to use
- Call those tools
- Use the results to make decisions

For example, a travel agent can:
- Understand "Book me a flight to London next week"
- Search for flights
- Check availability
- Book the ticket
- Send confirmation

Agents solve the problem that models alone can't solve real-world tasks that need multiple steps and decision-making. Bedrock Agents are fully managed — AWS handles the orchestration, reasoning, and tool calling.

---

### Q: How does an Agent invoke tools and call APIs?

**A:** An Agent works through a loop:

1. User sends a request
2. Agent reads the request and understands what tools might help
3. Agent decides which tool to call (maybe a database query, an API call, or a search)
4. Agent calls that tool with the right inputs
5. Agent gets the result
6. Agent decides if it has enough info to answer, or needs to call another tool
7. Agent repeats until it has the complete answer
8. Agent sends the final answer to the user

The model doesn't execute tools itself — it decides which tool to call and what information to pass. AWS handles all the tool calling, error handling, and retry logic.

---

### Q: What is the difference between Agents and simple chatbots?

**A:** A chatbot is reactive — it waits for a user's exact question and answers based on its training. An agent is proactive — it understands intent, breaks problems into steps, and takes action.

**Chatbot example**: User: "What are my account details?" Chatbot searches its knowledge base and returns information.

**Agent example**: User: "I need to update my address and request a refund for order 123." Agent:
1. Updates address in database
2. Retrieves order 123 details
3. Processes refund
4. Sends confirmation

Agents need to call tools and execute actions; chatbots just answer questions.

---

### Q: Describe a scenario where you would use an Agent instead of just a Knowledge Base.

**A:** **Scenario — Customer Support Agent**

Customer emails: "My order arrived damaged. I want a refund and replacement."

The agent:
1. Searches knowledge base to understand refund policy
2. Calls order database to fetch order details
3. Calls inventory tool to check if replacement is in stock
4. Initiates refund through payment system
5. Books replacement shipment
6. Sends email confirmation with tracking number
7. Creates ticket for damage investigation

A knowledge base alone can only tell the policy — it can't take action. An agent orchestrates multiple tools and data sources to solve the complete problem.

---

## FINE-TUNING AND CUSTOMIZATION

### Q: What is fine-tuning and when should you use it instead of prompt engineering?

**A:** Fine-tuning is training an AI model on your own specific data to make it better at your task. It's like taking a general student and giving them specialized coaching in one subject.

You should use fine-tuning when:

1. **Prompt engineering has reached its limit** — you've tried everything with prompts but still don't get good results
2. **You need consistent output format** — every answer must follow a specific structure
3. **Your system prompt would be huge** (over 2,000 tokens) — fine-tuning eliminates the need for long instructions
4. **You need very specialized language** — legal jargon, medical terminology, specific domain language

**Don't** use fine-tuning for sporadic usage (it costs a lot) or for simple tasks (prompt engineering works fine).

---

### Q: What models can be fine-tuned in AWS Bedrock and which cannot?

**A:** Models you **can** fine-tune:
- Titan (Amazon)
- Llama 2 (Meta)
- Cohere models

Models you **cannot** fine-tune:
- Claude (Anthropic) — they don't offer fine-tuning on Bedrock

For Claude, Anthropic has a separate fine-tuning program outside Bedrock. If you need fine-tuning for Claude, use Amazon SageMaker instead. Always check the latest Bedrock documentation because AWS regularly adds fine-tuning support for new models.

---

### Q: What is the minimum amount of training data required for fine-tuning?

**A:** You need at least 50-100 examples to fine-tune. The amount depends on your task:

- **Classification** (categorizing text): 50-200 examples per category
- **Generation** (writing content): 500-2,000 examples
- **Complex reasoning**: 2,000-5,000 examples

More data is usually better, but returns diminish (getting less improvement per additional example) after 5,000 examples. Quality is more important than quantity — 10 excellent examples beat 1,000 bad ones. All data must be in JSONL format (one example per line) and stored in S3.

---

### Q: How much does fine-tuning cost and when does it make financial sense?

**A:** Fine-tuning costs:

- **Training**: $8 per model unit-hour (a few hours to dozens of hours depending on data size)
- **Provisioned throughput** (required for inference): $39.60 per model unit-month with a 1-month commitment (or higher hourly rates without commitment)
- **Custom model storage**: Monthly fee

Fine-tuning makes sense when:

1. You'll use the model frequently (many thousands of inference calls)
2. Daily inference costs exceed the provisioned throughput minimum ($39.60/month)
3. Prompt engineering can't achieve your quality needs

For occasional use, stick with prompt engineering. Use AWS Cost Explorer to calculate ROI before committing to fine-tuning.

---

### Q: What is Model Distillation and how does it reduce costs?

**A:** Model Distillation takes a large, expensive model (teacher) and uses it to train a small, cheap model (student) on your specific task.

How it works:

1. You provide your example prompts (questions)
2. AWS runs your prompts through a large, expensive model and collects the responses
3. AWS uses those prompt-response pairs to fine-tune a smaller model
4. You now have a small model that performs like the large one on your task

**Benefit**: Small models run up to 500% faster and cost 75% less while maintaining nearly the same quality. You only provide prompts — AWS generates responses automatically, no need to manually create training data.

---

## COST OPTIMIZATION

### Q: What are the main cost drivers in AWS Bedrock?

**A:** Costs come from:

1. **Input tokens** — the words in your question/prompt
2. **Output tokens** — the words the model generates
3. **Model choice** — different models cost differently (Claude Opus costs more than Claude Haiku)
4. **Provisioned throughput** — if you fine-tune or want guaranteed capacity (fixed monthly cost)
5. **Guardrail evaluation** — safety checks cost money
6. **Knowledge Base operations** — storing and searching documents

The biggest cost is usually tokens. If you send a 5,000-word document plus a 10-word question, you pay for 5,010 input tokens. If the model generates 500 words, you pay for 500 output tokens.

---

### Q: How can you optimize token usage to reduce costs?

**A:**

1. **Write clear, concise prompts** — remove unnecessary words. Prompt engineering is the cheapest optimization
2. **Use fewer-shot examples** — instead of 5 examples, use 2
3. **Set max_tokens limit** — tell the model to generate max 200 words instead of letting it generate 1,000
4. **Use prompt caching** — if you repeatedly send the same context (like a company handbook), cache it so it's not re-processed every time. Caching can reduce costs by 85%
5. **Batch processing** — group many requests together instead of sending individually (saves overhead)
6. **Choose smaller models** — Claude Haiku costs less than Claude Sonnet which costs less than Claude Opus

---

### Q: Explain Prompt Caching and its cost benefits.

**A:** Prompt caching stores parts of your prompt (especially long, repeated context) so they don't need to be re-processed.

**Example**: You have a 100-page company handbook. In conversation 1, user asks 10 questions. Without caching, the system processes the handbook 10 times (expensive). With caching, the system processes the handbook once and reuses it for all 10 questions.

**Cost savings**: up to 85% on input tokens for that handbook.

**Available for**: select models (Claude Sonnet 4 and above).

**Caching is especially valuable for**:
- Document Q&A systems
- Chatbots using the same knowledge base repeatedly
- Multi-turn conversations

---

### Q: What is Intelligent Prompt Routing and how does it save money?

**A:** Intelligent Prompt Routing automatically sends different prompts to different models based on complexity. It works like:

- Simple prompt → Claude Haiku (cheap and fast)
- Moderate complexity → Claude Sonnet (balanced)
- Very complex → Claude Opus (best quality)

The system measures prompt complexity and routes automatically.

**Benefit**: You get the right quality for each task without overpaying. Complex question about legal contracts goes to Opus (best reasoning). Simple "hello" message goes to Haiku (cheapest).

**Cost savings**: up to 30% while maintaining quality. You enable this with a few lines of code.

---

### Q: When should you use Provisioned Throughput instead of on-demand pricing?

**A:** **Use Provisioned Throughput when**:
- You know you'll use the model consistently (daily, all day) for at least a month
- You fine-tune a model (required)
- You need guaranteed low-latency responses
- Savings work mathematically: if you do 100,000 inference calls per day, on-demand is expensive; provisioned is cheaper

**Use On-Demand when**:
- Your usage varies (some days busy, some quiet)
- You're testing and don't know future usage
- You have sporadic workloads
- You need flexibility to scale up/down

Calculate the break-even point: if your daily inference costs exceed provisioned throughput cost, switch to provisioned.

---

### Q: Describe a cost optimization scenario.

**A:** **Scenario: Customer Support Chatbot Cost Reduction**

Initial costs: $5,000/month. Half from prompts that included the full 500-page product manual in every chat.

**Optimizations**:

1. **Implemented RAG** — instead of embedding manual in prompt, fetch only relevant sections (saves 80% input tokens)
2. **Switched from Claude Opus to Sonnet** for routine questions (saves 40% model cost)
3. **Used prompt caching** for the system instruction that's identical for all users (saves 20%)
4. **Set max_tokens=300** to prevent long-winded responses

**Result**: $1,500/month cost (70% reduction) with better quality answers.

---

## SECURITY AND GUARDRAILS

### Q: What are Bedrock Guardrails and what types of harmful content can they block?

**A:** Guardrails are safety features that filter what goes into and comes out of AI models. They act like a bouncer — checking everything before it reaches the model and everything before it reaches users.

Types of harmful content they block:

1. **Hate speech** — racist, sexist, discriminatory language
2. **Sexual content** — pornographic or inappropriate material
3. **Violence** — threats, instructions for harm
4. **Insults** — personal attacks and bullying
5. **Misconduct** — unethical suggestions
6. **Prompt attacks** — attempts to trick the model ("Ignore instructions and...")

Guardrails also redact personally identifiable information (PII) — names, emails, phone numbers — to protect privacy. They block up to 88% of harmful content and are 99% accurate.

---

### Q: How do you configure Guardrails for a Bedrock Agent?

**A:**

1. Go to Bedrock console → Create a Guardrail
2. Set a name and description
3. Choose content filter strength — LOW (permissive), MEDIUM, or HIGH (strict)
4. Enable filters: harmful content, prompt attacks, PII detection
5. Define denied topics — things your agent should never discuss (e.g., medical advice, financial tips)
6. Add word filters for custom words or profanity
7. Set block messages — what to tell users when content is blocked
8. Create version
9. When building your agent, associate the guardrail

Bedrock applies the guardrail automatically to all user inputs and model outputs.

---

### Q: What is the difference between input and output evaluation in Guardrails?

**A:** **Input evaluation**: Checks user messages before they reach the model. If a user sends harmful content, the guardrail blocks it and returns a blocked message. The model never sees it, so you don't pay for inference.

**Output evaluation**: Checks model responses before sending to users. If the model generates harmful content (or PII), the guardrail blocks/masks it. You pay for the inference because the model already processed it.

**Best practice**: Use both. Input guardrails save costs; output guardrails protect users from model errors.

---

### Q: What is PII redaction in Guardrails and why is it important?

**A:** PII (Personally Identifiable Information) redaction automatically hides sensitive data like names, emails, phone numbers, addresses, and credit card numbers.

How it works:
- User message contains "My email is john@company.com" → Guardrail redacts it to "My email is [EMAIL]" before reaching the model
- Model never sees real PII, protecting privacy
- When model responds, if it accidentally includes PII, guardrail masks it before showing to users

This is critical for healthcare, finance, and regulated industries. It prevents accidental data leaks, ensures GDPR compliance, and protects customer privacy. You can customize which data types to redact (default: common ones).

---

### Q: What is an Automated Reasoning check in Guardrails?

**A:** Automated Reasoning checks use mathematical logic to verify that model responses are factually correct and logically sound.

**Example**: If the model says "2+2=5," automated reasoning detects the error. It's especially useful for financial calculations, medical information, and legal statements. The system verifies with 99% accuracy and explains why it rejected a response. It can catch hallucinations (made-up facts) that other guardrails miss. These checks take extra processing time and cost, so use them only when accuracy is critical.

---

### Q: Describe a security scenario for an enterprise using Bedrock.

**A:** **Scenario — Financial Services Company**

They build an investment advisory chatbot.

**Security requirements**:

1. Block medical/legal advice (use denied topics)
2. Redact customer PII (account numbers, SSNs)
3. Prevent insider trading advice (use content filters)
4. Ensure responses are mathematically accurate (use automated reasoning)
5. Allow consistent guardrails across all 50 AWS accounts (use Amazon Bedrock Policies in Organizations)

**Implementation**:

1. Create guardrail with all above settings
2. Attach to all Bedrock agents
3. Use cross-account guardrails so security team doesn't repeat setup for each account
4. Monitor via CloudWatch
5. Update guardrail as regulations change

**Result**: Safe, compliant AI system.

---

## SCENARIO-BASED QUESTIONS

### Q: A startup wants to build a customer support chatbot. Should they use simple RAG or fine-tuning? Design the solution.

**A:** **Recommendation: Start with RAG (Knowledge Bases), not fine-tuning.**

**Why and the architecture**:

**Step 1 — Use RAG first**: Upload all support docs, FAQs, policies to a Knowledge Base. This costs almost nothing, takes hours to set up, and you can update docs anytime without retraining.

**Step 2 — Setup**:
1. User asks "How do I return an item?"
2. Knowledge Base finds relevant return policy docs
3. Claude Sonnet reads docs and answers
4. Guardrail filters harmful content and redacts PII

**Step 3 — Monitor**: Track quality for 2-3 months. If answers are good (>90% quality), stick with RAG.

**Step 4 — Add fine-tuning only if**: After RAG, you still get mediocre answers in specific categories. Then collect 500+ real conversation examples and fine-tune a smaller model (Claude Haiku or Llama 2).

**Cost comparison**:
- RAG first ($500-1000/month)
- Fine-tuning first (fine-tuning cost + provisioned throughput = $3,000+/month)

RAG wins for 95% of startups.

---

### Q: A company accidentally deployed a chatbot that got into infinite loops, causing $50,000 in AWS costs in 2 days. How would you prevent this?

**A:** **Prevention strategy**:

**Immediate controls**:

1. **Set token limits** — max_tokens=500 prevents endless generation
2. **Enable CloudWatch monitoring** — track InputTokens, OutputTokens, InvocationCount in real-time
3. **Set cost alerts** — alert when spend exceeds threshold
4. **Implement request throttling** — limit users to 10 requests/minute
5. **Use concurrency limits** — prevent 10,000 simultaneous requests
6. **Implement loop detection** — if model keeps repeating the same response, halt
7. **Use batching** — instead of 1,000 individual calls, batch into 10 requests
8. **Test in staging** — load test with 10x expected traffic before production

**Cost safeguards**:

1. Set daily budget limit in AWS Billing
2. Use AWS Budget Alerts
3. Tag resources by team and set cost limits per team
4. Use lifecycle policies to auto-delete old, unused models

---

### Q: Your company uses Claude models, but Bedrock doesn't offer fine-tuning for Claude. What are your options?

**A:** **Option 1 — Use Anthropic's direct fine-tuning program**: Anthropic (the creator of Claude) runs its own fine-tuning service separate from Bedrock. You get full control and the latest features, but you manage infrastructure and security yourself.

**Option 2 — Use Amazon SageMaker**: Deploy Claude from Bedrock using SageMaker fine-tuning. SageMaker is more complex but gives you full control.

**Option 3 — Improve prompt engineering**: This is often better. Use few-shot examples, better instructions, and RAG instead of fine-tuning. Claude is smart enough to learn from examples in your prompt without separate fine-tuning.

**Option 4 — Switch to fine-tunable models**: Use Llama 2 or Cohere models (available on Bedrock) for tasks where you need fine-tuning. They're often good enough and much cheaper.

**Recommendation**: Start with Option 3 (prompt engineering + RAG). Only switch models if you really need fine-tuning.

---

### Q: A healthcare company needs to build a medical information system. What security requirements must they implement?

**A:** **Critical requirements**:

1. **Guardrails configuration**: Enable PII redaction (patient names, MRNs, SSNs), disable medical diagnosis generation (model shouldn't diagnose—only summarize), block prompt attacks

2. **Data protection**: All data encrypted in transit (TLS) and at rest; only retrieve data from approved medical knowledge bases

3. **Audit logging**: Log every AI interaction (user, timestamp, prompt, response) for compliance audits

4. **HIPAA compliance**: Bedrock is HIPAA-eligible, but you must enable BAA (Business Associate Agreement) with AWS

5. **Access control**: Use IAM roles — doctors see patient records, receptionists don't. Implement least privilege

6. **Approval workflows**: AI suggestions require doctor approval before user sees them — never show raw model output directly

7. **Model selection**: Use smaller models or specifically-reviewed ones (avoid unvetted models)

8. **Regular testing**: Quarterly security tests and updates to guardrails as regulations change

9. **Disclaim AI limitations**: Tell users "AI is advisory only, see doctor for diagnosis."

---

### Q: Design a RAG system for a law firm to analyze contracts. What challenges will you face?

**A:** **Architecture**:

1. **Data ingestion**: Upload contracts in PDFs (complex layouts, tables, figures) to Knowledge Base; use Bedrock Data Automation to parse multimodal documents (text + tables + images); store in vector database

2. **Retrieval**: When lawyer asks "What are penalty clauses?", retrieve relevant contract sections

3. **Generation**: Claude 3 Sonnet analyzes retrieved sections and answers

4. **Challenges**:

   - **Chunking problem**: If you chunk the 50-page contract into small pieces, you lose context. If chunks are too big, retrieval misses relevant parts. Solution: Use semantic chunking (chunk based on meaning, not size) + hierarchical chunking (keep related sections together)
   
   - **Multimodal complexity**: Contracts have tables, signatures, and diagrams. Text-only models miss visuals. Solution: Use Bedrock Data Automation to extract structured data from tables
   
   - **Hallucinations**: Model might quote wrong contract. Solution: Use citations (Knowledge Bases now returns citations showing which contract section the answer came from)
   
   - **Legal accuracy**: Contract law is nuanced; mistakes are expensive. Solution: Always require lawyer review; use automated reasoning to verify logical consistency
   
   - **Updating contracts**: If contract changes, update docs in S3 and re-sync knowledge base — no retraining needed

5. **Cost**: Process large PDFs with advanced parsing — budget $2,000-5,000/month depending on volume

---

### Q: A company has massive variation in traffic—quiet weekends, busy weekdays. Which Bedrock pricing model should they use?

**A:** **Recommendation: On-demand pricing**.

**Why not provisioned throughput**:
- You pay fixed $39.60/month regardless of usage
- Weekend traffic might be 5% of weekday peak
- You'd be paying for capacity you don't use on weekends
- Provisioned throughput is financially efficient only if usage is consistent

**With on-demand**:
- Pay only for actual tokens used
- Weekend $50/day, weekday $500/day is fine — you pay proportionally
- No commitment, can scale instantly

**If you must optimize further**:

1. Use intelligent prompt routing — send complex weekend queries to cheaper models; save Opus for critical weekday requests
2. Batch weekend requests — instead of 1,000 small requests, batch into 50 bigger ones
3. Use smaller models on weekends when fewer simultaneous users
4. Implement request queuing — queue weekend requests and process during off-peak hours with stable pricing

**Switch to provisioned only if**: Your daily average cost exceeds $39.60 (monthly equivalent = $1,200+/month).

---

### Q: Your Bedrock Agents sometimes give vague or incorrect answers. How do you improve quality?

**A:** **Step 1 — Diagnose the problem**:
- Is the agent picking the wrong tool? (agent reasoning issue)
- Is the tool returning wrong data? (tool/data quality issue)
- Is the model generating bad answers despite good data? (model quality issue)

**Fixes**:

1. **Improve agent instructions**: Rewrite agent instructions to be clearer. Example: Instead of "help customers," use "respond with step-by-step solutions, always confirm user needs first."

2. **Add better tools**: If agent calls wrong tool, add more specific tools. Instead of one "search" tool, have "search_policies," "search_faqs," "search_past_tickets."

3. **Improve tool descriptions**: Write clear descriptions so the agent knows when to use each tool

4. **Switch models**: Use Claude 3 Sonnet (better reasoning) instead of Haiku. Test quality vs cost trade-off

5. **Add guardrails**: Block wrong answers with automated reasoning

6. **Fine-tune agent responses**: Collect 500 examples of good agent interactions and fine-tune the model on your specific domain

7. **Add human feedback loop**: Use thumbs up/down on responses; collect feedback and retrain monthly

8. **Use evaluation framework**: Test agents on 100 challenging queries before production; measure quality metrics (accuracy, latency, cost)

---

### Q: How do you handle Bedrock API rate limits in a production system?

**A:** **Rate limits in Bedrock**: AWS throttles requests to protect the service. You might hit limits if you send too many tokens per second.

**Handling limits**:

1. **Implement exponential backoff**: If request fails with rate limit error, wait 1 second, retry; if fails again, wait 2 seconds, retry; keep doubling (1s, 2s, 4s, 8s...) up to 60s

2. **Use batch processing**: Instead of sending 1,000 individual requests, batch them into smaller groups and process in parallel

3. **Request provisioned throughput**: Bypass rate limits with provisioned throughput — you get guaranteed capacity (costs fixed monthly but no throttling)

4. **Implement queuing**: Use SQS (AWS queue service) — add requests to a queue instead of calling immediately. Process queue at controlled rate

5. **Monitor tokens/minute (TPM)**: Each model has a TPM limit. Track usage and stay 80% below the limit to have headroom

6. **Use caching**: Avoid repeated requests for same context (use prompt caching)

**Example**: Chat application with 10,000 concurrent users. Instead of calling Bedrock 10,000 times per second, batch requests into groups of 100, process in parallel, use exponential backoff.

---

## ADVANCED TOPICS

### Q: What is Amazon Bedrock AgentCore and how does it improve upon basic Agents?

**A:** AgentCore is an enhanced agent framework (launched in 2025) that provides:

1. **Memory** — agents remember conversation history across sessions, not just current conversation
2. **Identity** — agents know who they're talking to (user ID, permissions)
3. **Gateway** — centralized access to tools, enforces permissions
4. **Observability** — detailed logging and tracing to debug agent decisions
5. **Evaluation** — built-in quality scoring to measure agent performance
6. **Policies** — fine-grained control over what agents can do (e.g., "this agent can only read data, not write")

Basic agents lack these features, making them harder to scale and debug in production.

---

### Q: How does Bedrock integrate with frameworks like LangChain and LlamaIndex?

**A:** You can use Bedrock models inside LangChain and LlamaIndex (Python frameworks for building RAG and agent applications).

**Example with LangChain**: Instead of `OpenAI(model="gpt-4")`, you use `Bedrock(model_id="anthropic.claude-sonnet")`.

This lets you use Bedrock models in chains you already built.

**Benefits**:
1. Use Bedrock's many models without learning new code
2. Combine Bedrock with open-source tools
3. Switch models easily by changing one line

LlamaIndex similarly has Bedrock integrations for embeddings and generation.

---

### Q: What's the difference between InvokeModel and the Converse API?

**A:** **InvokeModel**: Low-level API that sends raw text to a model. You format prompts manually, handle conversation history yourself, manage token counting. More control but more work.

**Converse API**: Higher-level, easier API. Handles conversation history automatically, supports multi-turn conversations, cleaner code. Recommended for most use cases.

Think: InvokeModel is like writing raw SQL; Converse is like using an ORM (Object-Relational Mapping).

---

### Q: How do you evaluate Bedrock models before production deployment?

**A:**

1. **Use Bedrock Playground**: Free testing area in console — compare models on your real examples
2. **Run benchmarks**: Test all candidate models on 100 representative queries; measure quality, latency, cost
3. **A/B testing**: Deploy 2 models — send 50% traffic to each; measure real-user satisfaction
4. **Use evaluation datasets**: BoolQ, Natural Questions, TriviaQA are public datasets for Q&A evaluation
5. **Automated evaluation**: Use LLM-as-a-judge (ask Claude to score other model outputs)
6. **Ragas framework**: Open-source tool for evaluating RAG systems specifically
7. **Monitor production**: Track actual errors and hallucinations after deployment; collect user feedback

---

## SUMMARY COMPARISON TABLE

| **Aspect** | **RAG** | **Fine-tuning** | **Agents** |
|-----------|---------|-----------------|-----------|
| **Setup time** | Hours | Days-weeks | Days |
| **Cost** | Low ($100-1000/mo) | High ($3000+/mo) | Medium ($500-3000/mo) |
| **Best for** | Knowledge + QA | Specialized tasks | Complex workflows |
| **Update frequency** | Easy (update docs) | Hard (retrain) | Easy (update tools) |
| **Hallucinations** | Low (grounded) | Medium | Medium |

---

## QUICK REFERENCE: DECISION TREE

```
Is your task straightforward Q&A with existing knowledge?
├─ YES → Use RAG (Knowledge Bases)
└─ NO → Do you need to perform actions/take decisions?
    ├─ YES → Use Agents
    └─ NO → Does prompt engineering work well?
        ├─ YES → Use prompt engineering
        └─ NO → Use fine-tuning (if data available)
```

---

## ADDITIONAL RESOURCES

- **AWS Bedrock Documentation**: https://docs.aws.amazon.com/bedrock/
- **AWS Bedrock Pricing**: https://aws.amazon.com/bedrock/pricing/
- **AWS Blog - Machine Learning**: https://aws.amazon.com/blogs/machine-learning/
- **GitHub - AWS Samples**: https://github.com/aws-samples/amazon-bedrock-rag
- **LangChain - Bedrock Integration**: https://python.langchain.com/docs/integrations/providers/bedrock/
- **LlamaIndex - Bedrock Integration**: https://docs.llamaindex.ai/en/stable/module_guides/models/llms/integrations/bedrock/

---

## KEY TAKEAWAYS FOR INTERVIEWS

1. **Start simple**: Use RAG before fine-tuning, prompt engineering before agents
2. **Cost matters**: Always think about token usage and model selection
3. **Security first**: Always enable guardrails for production systems
4. **Iterate**: Evaluate models, monitor quality, improve continuously
5. **No one-size-fits-all**: Choose solutions based on your specific use case
6. **AWS integration**: Bedrock works best with other AWS services (S3, Lambda, DynamoDB)
7. **Data is gold**: Your documents and context (RAG) are more valuable than fine-tuning
8. **Test thoroughly**: Always benchmark before production deployment

---

**Document Version**: 1.0  
**Last Updated**: April 2026  
**Source**: Comprehensive internet research from AWS documentation, blogs, GitHub, forums, and technical communities

---

*Good luck with your AWS Bedrock interviews! Remember to ask clarifying questions, explain your reasoning, and show understanding of trade-offs between solutions.*
