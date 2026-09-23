# UI Context

## App Type
Single-page Streamlit app (can expand to multi-page later if needed).

## Screens / Sections

### 1. Input Form
Fields matching the dataset's applicant features, e.g.:
- Number of dependents
- Education (Graduate / Not Graduate)
- Self-employed (Yes/No)
- Annual income
- Loan amount requested
- Loan term
- CIBIL score
- Asset values (residential, commercial, luxury, bank)

Use Streamlit widgets — `st.number_input`, `st.selectbox`, `st.slider` as appropriate. Keep it one form with a submit button, not scattered auto-updating fields.

### 2. Output Section
Shown after submission:
- **Approval result:** Approved / Rejected, with the model's confidence (probability)
- **Risk tier:** Low / Medium / High, with a short explanation of which factors drove it (e.g., "Low CIBIL score relative to loan amount")

### 3. Insights Section (optional, if time allows)
- A small chart (bar / feature importance) showing which features most influenced the decision
- Dataset-level stats (e.g., approval rate by property area) for context

## Style Notes
- Keep it functional over polished — this is a course project demo, not a production product
- Use `st.columns` to place the form and result side-by-side, or stacked if simpler
- Avoid page reloads mid-flow — one form, one result block
