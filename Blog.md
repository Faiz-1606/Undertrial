**Reasoning Over Prediction: An RL Approach to Bail Decisions**

Three minutes should never decide a life. Yet in countless Indian bail hearings, that is exactly the time a person gets. There are many places where every second feels heavy; time doesn’t just pass it presses such as in a battlefield, an emergency room and often overlooked, the Indian court room on a bail hearing day.

The files don’t stop coming. One after another, names are called. Lawyers step forward, speak fast, compress arguments into fragments. The judge listens, scans, decides.

Three minutes. That’s all it takes to decide whether someone walks out of that courtroom or goes back to a cell, waiting. Not convicted. Not proven guilty. Just waiting.

Now this very process repeats 80 to 100 times a day. It’s not a flaw in the system. But the system being stretched to its absolute limit with visible consequences. Today, nearly 76% of India’s 5.7 lakh prisoners are undertrials people who have not been convicted, but are still behind bars. Many of whom cannot afford strong legal representation. Many wait not because their case is strong, but because the system does not have the time to fully see them.

Amidst this we found ourselves asking questions that almost felt impossible;

What if we could build a system that never rushed?  
That reads every document carefully?  
That applied the law consistently?  
That learned from past decisions but at the same time didn’t blindly repeat their biases?

Initially, it sounded unrealistic. Justice is human it isn’t any formula. It demands empathy, context, nuance that no system could replicate. And that’s where we drew the line.

We didn’t try to build a system that decides bail but that could think alongside. A system that slows things down where it matters reading every page, tracing every legal thread, laying out reasoning with care and consistency.

That’s how UndertriAI came to life.

We didn’t build a chatbot for bail, but a closed-loop world where an LLM learns to behave like a careful legal assistant using tools, tracking context, and improving from feedback. It is an OpenEnv based reinforcement learning environment focused not on predicting outcomes, but on training process.

In the beginning, it struggles it reads charge sheets but misses key details, weighs arguments inconsistently, and produces incomplete reasoning. That is intentional the system is meant to learn. Each case becomes an interaction where it must read documents, check statutory eligibility, reference precedents, and evaluate factors like custody duration, flight risk, and parity before writing a structured bail memo. This memo is then evaluated against real High Court decisions. Strong reasoning is rewarded; flawed logic or bias is penalized. Over time, the system changes. It slows down its thinking, becomes more structured, and learns how conclusions are built.

Each episode begins with a case file where the agent must interpret facts, arguments, and incomplete context, balancing multiple signals before forming a recommendation. Trained using GRPO through the TRL stack, the focus is on improving behaviour, performing checks, using tools effectively, and reasoning clearly without shortcuts. A multi-stage curriculum ensures the agent progresses from simpler to more complex cases, learning through adaptation rather than repetition.

To know more about how this reasoning unfolds in practice, you can explore it here:

[Live environment](https://huggingface.co/spaces/Draken1606/undertrial-ai)

[Source code](https://github.com/Faiz-1606/Undertrial)

In a system constrained by time, UndertriAI helps make thinking clearer, more consistent, and more visible, ensuring better-informed human decisions. And that is the real goal. Not to automate justice, not to replace judgment, but to make reasoning harder to ignore. Because justice was never meant to be fast it was meant to be fair. And if a system can help us move even slightly closer to that, even within three minutes, then perhaps attempting the impossible was worth it.