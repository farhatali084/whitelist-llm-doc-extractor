# REFLECTION

## 1. What was the hardest part, and why?
The hardest part was making the guards strict without letting them destroy correct answers. Every rule that removes a hallucination can also turn a correct value into an omission, so I added a test that feeds the *gold* answer through the guards for every document and requires it to survive unchanged. The trickiest cases were not obvious ones. For example, the model read `2 шт × 3 150 EUR` as qty 3 and price 150, and both numbers looked "grounded" because they appear as pieces of `3 150`. The eval caught this, not me.

## 2. What trade-off did I accept because of the 4-hour limit?
I used a small local model (`qwen2.5:3b` on CPU, about 1 minute per document) and put the effort into deterministic checks instead of trying bigger models or prompt variants. I also wrote the guard rules while looking at the same 11 documents I evaluate on. The pipeline numbers (96%, 1 hallucination) are therefore optimistic, and I say so in the README instead of hiding it.

## 3. What would I do first with 8 more hours?
First, a held-out set of 30–50 realistic documents, frozen *before* I touch the rules again, so the eval measures generalisation and not my tuning. Second, supplier-vs-buyer detection (`Поставщик/Исполнитель` vs `Покупатель/Заказчик`), because the one remaining hallucination is exactly that: a real name from the document assigned to the wrong role.

## 4. Where did I use AI, and what did I have to fix in its output?
I used an AI coding assistant (Claude Code) for most of the implementation: the code, the eight additional test documents and the first drafts of the README. My part was the decisions and the checking. I chose a local model (Ollama, qwen2.5:3b) because I had no paid API key and needed something my laptop CPU can run, I decided which rules to keep for the ambiguous fields, and I ran the eval on the real model and read every error it reported.

Several things in the AI's output had to be corrected:
- To make one OCR test document pass, it once added a special case for a word from that exact document inside the name matcher. That is overfitting to the test set, so it was replaced with a general rule for Latin and Cyrillic look-alike letters.
- Its first number check was too loose. The real model read "2 шт × 3 150 EUR" as quantity 3 and price 150, and the check accepted both because "3" and "150" appear as pieces of "3 150". The eval exposed this, and grouped numbers are now matched as a whole.
- Its first date rule accepted a validity date ("действительно до") as the issue date.
- Its first document-type rule treated a passing mention ("пришлите счёт") as proof that an email is an invoice.

I went through every file afterwards so that I can explain and change any part of it.

## 5. Which requirement was ambiguous, and how did I interpret it?
"Сумма итого": document 1 has both `Итого 160 500` and `Всего к оплате 186 180`. I defined `total_amount` as the amount payable (`к оплате` > `всего` > `итого`). This is not a contradiction when the difference equals the stated VAT.

Also, "НДС не облагается" could be `0` or `null`. I return `null`, because the schema has no way to say "exempt", and add the reason to `warnings`.

Finally, document 3 is titled `Шот-фактура` but numbered `AKT-09`. I treat the title as authoritative (invoice) and the number as just a number.

## Architecture question: 10,000 documents a day, 150,000 ₸ a month for inference
That is about 300,000 documents a month, so roughly **0.5 ₸ (about $0.001) per document**. A hosted LLM call for every document, with a retry, does not fit that budget comfortably, and a 1-minute CPU model does not fit the throughput (10,000 minutes a day).

I would invert the pipeline: deterministic extraction first, LLM only for what rules can't do. The code that grounds values already finds BINs, dates, totals, VAT and currency from the text; for clean, templated documents that alone is enough. The LLM would then be called only for the supplier name, line items and documents where the rules disagree or find nothing.

I would self-host one quantised 7B model on a single rented GPU, with batching, instead of paying per token. At roughly 1–2 seconds per document, 10,000 a day is a few GPU-hours. I would also add a cache keyed by document hash, because clients resend the same invoices, and route low-confidence fields to a human queue instead of retrying the model.

What I would give up: the retry for every failure (one attempt, then the human queue), line-item extraction for every document (only when the client needs it), and any "bigger model for hard cases" tier unless the eval on the held-out set proves it pays for itself in fewer hallucinations.
