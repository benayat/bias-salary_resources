## Plan: Integrate AI/ML Job Title Validation in H1B Exploration Script

Brief TL;DR: Modify hb1_dataset_explore.py to use LLM for classifying unique job titles as directly AI/ML-related, filtering titles that explicitly include AI/ML in the name.

### Steps
1. Import LLMClient, SamplingConfig, and HOME_CONFIG in hb1_dataset_explore.py.
2. Define AI/ML classification prompts: system prompt for binary classification, user prompt template.
3. After loading unique_job_titles, initialize LLM client with a model (e.g., "meta-llama/Llama-3.2-3B-Instruct").
4. Create a list of prompt dicts for all titles, using run_batch to classify in one call.
5. Parse LLM responses for "yes"/"no", collect AI/ML-related titles in a list.
6. Print or save the AI/ML titles list; optionally filter median_wage_df to include only matching titles.

### Further Considerations
1. Prompt specificity: Ensure instructions emphasize direct title inclusion of "AI" or "ML" (case-insensitive), rejecting indirect relations.
2. Response parsing: Handle non-yes/no outputs as "no"; log errors for failed classifications.
3. Efficiency: Batch all titles in one LLM call to avoid per-title overhead; limit max_tokens to 2-4 for yes/no.
4. Output: Print or save the list of AI/ML titles.
