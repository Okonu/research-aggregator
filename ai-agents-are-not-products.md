# AI Agents Are Not Products: Embedding Intelligence Into Business Operations

**Ian Okonu**
April 12, 2026 · 16 min read

---

## On This Page

- Abstract
- 1. Introduction: Two Versions of the Same Operation
- 2. What an Agent Actually Does
- 3. Three Ways Businesses Are Getting This Wrong
  - 3.1 The Interface Trap
  - 3.2 Automating the Wrong Layer
  - 3.3 Measuring the Wrong Outcomes
- 4. How to Know If Your Process Is Ready
- 5. What Good Looks Like: Real Deployments
- 6. A Practical Framework for Embedding Agents
  - Component 1: Start With the Decision, Not the Technology
  - Component 2: Build the Readiness Filter Before the Tool Selection
  - Component 3: Define the Human Role Explicitly
  - Component 4: Measure Business Process Outcomes
- 7. Where to Start
- 8. Conclusion: The Operations Question
- References

---

## Abstract

AI agent products are proliferating faster than the operational frameworks needed to deploy them well. Most enterprise and SME deployments fail not because the underlying models are incapable, but because the surrounding design is wrong: teams build chat interfaces instead of decision systems, automate outputs instead of decisions, and measure task completion instead of business process outcomes. This article identifies three structural failure modes in current agent deployments, proposes four filters for evaluating process readiness, and presents a practical framework for embedding AI agents into business operations in a way that produces measurable outcomes. Real deployments at Morgan Stanley, NVIDIA, and Adobe illustrate what the pattern looks like in production. The central argument is that embedding intelligence into operations is an operations design problem, not a technology selection problem.

---

## 1. Introduction: Two Versions of the Same Operation

There is a version of customer onboarding that takes three days. A human reviews the application, pulls account history, checks documents, routes approvals, and eventually reaches a decision. The process is familiar, defensible, and slow.

There is another version that takes under fifteen minutes. An agent receives the application, identifies the customer record, checks the internal policy conditions, drafts a personalised response, escalates the edge cases, and logs the interaction — without a human initiating each step. The difference is not which software vendor the company is using. The difference is how the work was designed.

In 2024, Klarna announced that an AI agent had handled 2.3 million customer service conversations in a single month — the equivalent of 700 full-time staff — cutting average resolution time from 11 minutes to 2 minutes. The headline was everywhere. What got less coverage: by mid-2025, Klarna's CEO was publicly saying the company had over-rotated on AI and needed to bring more humans back into operations. [1]

Both things are true. The agent worked. And the deployment created problems the business did not anticipate — gaps in relationship management, edge cases the system handled badly, trust issues with customers who felt they were talking to a wall.

The lesson from Klarna is not that AI agents do not work. It is that plugging an agent product into operations without redesigning the process around it will produce partial wins and expensive surprises.

This is the gap that most of the current AI agent conversation does not address. The market is flooded with products — every software vendor from the largest enterprise platforms to two-person startups has added "AI agents" to their pitch. But buying an agent product and embedding intelligence into your operations are two entirely different things. One produces a demo that impresses in a board presentation. The other actually changes how work gets done.

This article is about the second one.

---

## 2. What an Agent Actually Does

Before examining where deployments go wrong, it is worth being precise about what distinguishes an agent from the AI tools most organisations are already using.

Strip away the terminology and an AI agent is a system capable of three things a standard chatbot or AI assistant cannot do:

**It takes actions, not just answers.** A regular AI assistant tells you that an invoice is overdue. An agent identifies the invoice, looks up the vendor contact, drafts a follow-up, sends it, and logs the interaction — without a human initiating each step.

**It works across multiple steps.** A single AI query is one input, one output. An agent runs a sequence: check the data, check the policy, make the decision, trigger the next step. Each action informs the next.

**It operates inside your systems.** The agent does not just generate text. It reads from and writes to your CRM, ERP, ticketing system, or database. Its outputs land in the real world — approved, rejected, escalated, created, updated.

That third property is what makes agents commercially significant and operationally risky at the same time. When an agent can actually act on your systems, getting the design right matters far more than when it is just generating a summary for a human to read and interpret.

The research framing (Dumas et al., 2026) situates this as a new wave of business process management: where previous automation technologies — RPA, workflow engines — moved fixed steps faster, agentic systems can reason about which steps to take, in what order, given the specific context of each case. [2] That distinction is the source of their value, and their risk.

| Dimension | Chatbot / AI Assistant | Embedded Agent System |
|---|---|---|
| Output type | Text response | System action |
| Steps | Single query | Multi-step sequence |
| System access | Read-only or none | Read and write |
| Human role | Reads output, then acts | Reviews exceptions |
| Design question | What can the AI answer? | What decision should it own? |
| Success metric | Response quality | Business process outcome |

---

## 3. Three Ways Businesses Are Getting This Wrong

Across enterprise and SME deployments, the same failure patterns recur. They are not model problems. They are design problems.

### 3.1 The Interface Trap

The most common implementation pattern: a business connects an AI to a knowledge base, puts a chat interface on it, and calls it an agent. Employees or customers type questions. The AI answers them. The team reports that the assistant is well-received.

This is a knowledge access tool. It has value. It is not what we mean by embedding intelligence into operations.

The tell is simple: if a human still has to read the AI's output and then go do something with it, the AI is a research assistant — not an operational system. The work has not been reduced; a step has been added. Productivity tools reduce friction. Embedded agents eliminate steps entirely.

The interface trap is partially a vendor problem. Most commercial agent products are designed around a chat interface because that is what is easiest to demo, easiest to deploy, and easiest to explain to a procurement committee. The interface is the product. What the business actually needed — a system that acts on its workflows — was never part of the design.

### 3.2 Automating the Wrong Layer

Some organisations go further. They give the AI tools, let it take actions, and deploy it. But they automate the output layer — sending a report, filling a form, generating a document — rather than the decision layer: should this invoice be approved, should this customer be escalated, should this lead be prioritised.

Output automation saves time. Decision automation transforms throughput.

A system that auto-generates a credit memo saves a clerk 15 minutes. A system that decides which accounts qualify for extended payment terms, and at what limits, is running part of the finance function. The operational impact of the second is an order of magnitude larger than the first.

The McKinsey Global Institute estimates that 60–70% of current work activities could be automated with existing technology. [3] The majority of that opportunity is not in the output layer. It is in the decision layer — the judgements that currently require a trained person, a policy reference, and access to live data. That is precisely what agentic systems are capable of, and precisely where most deployments are not pointed.

### 3.3 Measuring the Wrong Outcomes

Teams deploy agents and measure how often the agent completes tasks and how satisfied users are with its responses. These feel like the right questions. They are not.

The right questions are process questions: how has the workload on the operations team changed? How has the error rate moved? How much faster is the process end-to-end? What has it cost to resolve the exceptions the agent created?

A customer service agent with a 95% task completion rate sounds like success. If the 5% it cannot handle are the highest-value cases, and those customers are now sitting in a longer queue because the agent consumed the team's attention, the system has improved a metric while degrading the actual customer experience. Klarna learned this publicly. Many other organisations are learning it quietly.

---

## 4. How to Know If Your Process Is Ready

Not every process is worth automating with agents. The vendor conversation tends to start with capability questions — what can the product do, what does it integrate with — before the more fundamental questions have been answered. Before evaluating any tool, run the target process through four filters.

**Volume and repetition.** Does this process occur frequently enough that automation pays back the setup cost? A process that runs ten times a week is a strong candidate. One that runs twice a quarter is not — unless each instance is very high-stakes. The economics of agent deployment require sufficient volume for the system to pay for itself and to accumulate enough decision history to be monitored and improved.

**Definability.** Can you describe exactly what a good decision looks like? What information does it require? What are the rules? If three experienced members of your team would handle the same case the same way 80% of the time, the decision is definable enough to automate. If the answer is that it depends on contextual factors that are hard to articulate, automation will underperform. Agents are not good at making judgements that humans themselves cannot consistently describe.

**Data accessibility.** Does the information the agent needs exist in systems it can access? An invoice approval agent needs invoice data, vendor history, and budget information. If those live in different systems with no integration, the first problem to solve is data access, not agent design. The most common reason agentic deployments underperform is not model quality — it is that the agent cannot reach the information it needs to reason well.

**Consequence of error.** What happens when the system gets it wrong? Low-consequence errors — incorrectly categorised a support ticket, suggested the wrong internal document — are acceptable in automated systems. High-consequence errors — denied a loan incorrectly, sent customer data to the wrong party, approved a fraudulent payment — require human oversight in the loop. The agent can assist the decision; it should not own it unilaterally when the cost of being wrong is severe.

---

## 5. What Good Looks Like: Real Deployments

The following cases are not proof-of-concept demonstrations. They are production deployments measured against real business metrics.

**Morgan Stanley** deployed an AI system on top of their wealth management knowledge base — 100,000 documents covering research, products, and procedures — that financial advisors can query during client meetings. The output is not a document link; it is a synthesised response with citations. Reported adoption reached 98% of their financial advisors. [4] The reason it works: the agent does not replace the advisor's judgment. It reduces the time the advisor spends retrieving information, so more time goes to the judgment itself. The human role is preserved and improved, not replaced.

**NVIDIA internally** runs an AI knowledge assistant — NVInfo — serving over 30,000 employees. It handles technical support queries, HR policy questions, and IT requests at scale. [5] What is notable about their implementation is the feedback architecture: cases where humans override the agent's output are logged and used to improve the system continuously. Disagreements between agent and human are treated as data, not failures. The agent improves not through periodic retraining but through the accumulated signal of every case that did not go as expected.

**Adobe's engineering operations** team deployed an alert triage agent across their e-commerce infrastructure. Production alert systems generate hundreds of notifications daily; most are noise. The agent classifies, correlates, and prioritises alerts before they reach on-call engineers. [6] Mean time to resolution dropped, and the cognitive cost of evaluating high volumes of low-signal alerts decreased. This is an agent handling decision volume at a speed and consistency no human team could match — not replacing human judgment, but filtering so that human attention goes where it matters.

What all three cases share: the agent is inside a real workflow, it is measured against a real business metric, and there is a defined and explicit role for humans when the agent reaches the boundary of its competence.

---

## 6. A Practical Framework for Embedding Agents

The following framework is not a product evaluation guide. It is a design approach — four components that, taken together, describe what an embedded agent system requires to produce business outcomes rather than demo results.

| Layer | Common Approach | Embedding Requirement |
|---|---|---|
| Design entry point | Tool selection first | Decision definition first |
| Process readiness | Assumed | Tested against four filters |
| Human role | Unspecified | Explicit, with defined triggers |
| Success measurement | Task completion / UX score | Business process KPI delta |

### Component 1: Start With the Decision, Not the Technology

Before selecting a tool, define the decision. Not the task, not the capability, not the use case — the specific decision: who makes it today, what information it requires, what the rules are, what happens when it is wrong.

This sounds obvious. It is almost never done first. Most agent deployments begin with a vendor pitch or a platform evaluation. The decision framing comes later, if at all, and by that point the architectural choices have already been made.

A decision that is well-defined before design begins produces a system with clear success criteria, a natural escalation boundary, and an audit trail that maps to the decision's accountability chain. A decision that is vaguely framed produces a system that is difficult to evaluate, impossible to improve systematically, and expensive to defend when something goes wrong.

### Component 2: Build the Readiness Filter Before the Tool Selection

Run the target process through the four filters described in section 4 — volume, definability, data accessibility, consequence of error — before engaging with any vendor. This is a one-page internal exercise, not an extended assessment. Its purpose is to identify whether the problem is agent-ready, and if not, what needs to be addressed first.

Many processes that appear agent-ready fail on data accessibility. The decision is well-defined, the volume justifies automation, but the data sits in systems that are not connected. The right investment in that case is integration infrastructure, not an agent platform. Vendors will not tell you this. The readiness filter will.

### Component 3: Define the Human Role Explicitly

Every embedded agent system needs a documented answer to three questions about human involvement: when does the agent escalate to a human, what does the human do with what the agent has prepared, and who is accountable when the agent's decision turns out to be wrong.

The escalation condition should be specific and codified — not "when the agent is unsure" but "when aggregate confidence falls below 0.75, or when transaction value exceeds $40,000, or when an identity conflict cannot be resolved." These thresholds define the boundary between autonomous execution and human oversight. They are the most consequential design decisions in an agentic deployment, and they should be treated as governed artefacts — documented, version-controlled, and reviewed when the business process changes.

The accountability question is not technical. It is governance. When an agentic system makes a decision that turns out to be wrong, someone needs to own that outcome. That person should be identified before the system goes live, not after the first adverse outcome.

### Component 4: Measure Business Process Outcomes

Replace the standard agent metrics with the process metrics that existed before the agent was deployed.

| Replace this | With this |
|---|---|
| Task completion rate | Straight-through processing rate |
| User satisfaction score | Override rate (agent decisions reversed by humans) |
| Response latency | End-to-end process cycle time |
| LLM benchmark scores | Downstream error rate |
| — | Workload delta (FTE-hours per unit before vs. after) |

The override rate deserves particular attention. If humans are reversing 30% of the agent's decisions, the system is not saving time — it is creating review work. If the override rate falls to near zero, there may be a rubber-stamp problem: humans approving without genuinely reviewing. A well-calibrated embedded agent typically sees override rates between 5% and 15%, concentrated in the genuinely ambiguous cases that escalation policies are designed to catch.

---

## 7. Where to Start

For organisations that have not yet run a meaningful agent deployment, the most reliable starting point is a process that is high volume, low stakes per instance, and already well-defined. Invoice processing, first-response customer support, employee IT requests, document classification, meeting notes with action items — these are the processes where agents have a strong track record and where the cost of errors is manageable while the team learns how the system behaves.

The goal of a first deployment is not maximum efficiency. It is operational learning: how does the team interact with agent-generated decisions? What does the override rate look like at different confidence thresholds? Where does the agent underperform, and which data gaps are driving that underperformance? That learning shapes the architecture of the higher-stakes, higher-value deployments that come next.

Organisations that start with the hardest problems — fully automated credit decisions, end-to-end compliance review, customer-facing agents for sensitive service interactions — almost always build something they have to walk back. Start where the feedback loop is fast and the blast radius is small. Use what you learn there to build toward the complex.

---

## 8. Conclusion: The Operations Question

The adoption curve is not waiting. Research from McKinsey indicates that 78% of organisations are now using AI in at least one business function, up from 55% a year prior — and the fastest-growing category is autonomous workflow agents. [3] The competitive pressure to deploy is real, and so is the operational pressure to show returns on the investment.

The companies that will use agents well are not the ones that deployed the most products. They are the ones that invested in understanding their processes deeply enough to know exactly where a machine can own a decision, where it should support one, and where it should stay out of the way entirely.

That distinction is not a technology question. It is an operations question. Getting it right means starting with the decision, not the demo. It means running the readiness filter before signing a contract. It means defining the human role before the system goes live. And it means measuring the outcomes that matter to the business, not the outcomes that are easy to instrument.

Get the operations right, and the technology choice becomes almost secondary.

The practitioners making architectural decisions right now — at digital lenders, at logistics companies, at professional services firms — are defining how this generation of AI systems integrates with human work. Those decisions belong to the people designing the processes, not the people selling the products.

---

## References

*Papers and industry sources retrieved via the Research Aggregator built for multi-source academic and practitioner search. Source APIs: arXiv, Semantic Scholar, OpenAlex, CrossRef.*

[1] Leswing, K. "Klarna CEO says the company over-rotated on AI and now wants to hire more workers." CNBC, September 2025.

[2] Dumas, M., Milani, F., & Chapela-Campa, D. (2026). "Agentic Business Process Management Systems." arXiv:2601.18833. Frames agentic AI within the evolution of BPM — situates this generation of automation tools within the broader history of how organisations have used technology to run processes.

[3] McKinsey Global Institute. "The State of AI: How Organizations Are Rewiring to Capture Value." 2025. Annual survey tracking enterprise AI adoption rates and function-level deployment patterns.

[4] Somosi, A. "Morgan Stanley's AI assistant for financial advisors gets an upgrade." Reuters, October 2024. Documents the wealth management AI deployment, adoption metrics, and advisor use patterns.

[5] Shukla, A., Knowles, S., & Madugula, M. (2025). "Adaptive Data Flywheel: Applying MAPE Control Loops to AI Agent Improvement." arXiv:2510.27051. Documents NVIDIA's NVInfo internal deployment at scale (30,000+ employees), with focus on the continuous improvement architecture that uses human overrides as training signal.

[6] Bharadwaj, A. & Tu, K. (2026). "Agentic Observability: Automated Alert Triage for Adobe E-Commerce." arXiv:2602.02585. Case study of Adobe's production deployment of an alert triage agent — one of the clearest published examples of agents handling decision volume at scale.

[7] Kandogan, E., Bhutani, N., & Zhang, D. (2025). "Orchestrating Agents and Data for Enterprise: A Blueprint Architecture for Compound AI." arXiv:2504.08148. Addresses integration challenges enterprises face when deploying AI agents into existing infrastructure — relevant to the data accessibility filter discussed in section 4.

[8] Krishnan, N. (2025). "AI Agents: Evolution, Architecture, and Real-World Applications." arXiv:2503.12687. Reviews the progression from rule-based to LLM-powered agents, with emphasis on the gap between architectural capability and operational deployment reality.
