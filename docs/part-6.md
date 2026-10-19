# Part 6: The apply assistant (agent): what we built and why

6a = the brain (verify.py, agent_prompts.py, agent.py). 6b = the screen (ui_agent.py, the new tab in app.py).

(Rewrite in your own words. Questions to answer:)
- What makes this an AGENT and not a fixed pipeline? (state, conditional edge that loops back, two human interrupts)
- Why two layers of fact-checking? What does each catch that the other can't?
- Why is the deterministic layer limited to numbers and concrete tools? What does it miss? (plain-word claims like "led")
- Why does the retry only redo the FAILED bullets, and why is it capped at one retry? (cost, no endless loops)
- Why do nodes that call interrupt() contain no model calls? (they re-run from the top on resume)
- Why does the state hold dicts, and the profile/vector store live outside it? (clean checkpointing)
- Why does a missing AI-reviewer verdict count as a failure? ("fail closed")
- Why fall back to a template email instead of sending a risky one? How do you know the template is safe?
- How does a visitor's own API key reach code running inside the graph? (ContextVar copied into worker threads; tested)
- What would you add next? (LangSmith tracing, evaluation set of JDs, a judge model different from the writer)
