
### File Descriptions

- **components/**  
  Contains modular AzureML pipeline components:
  - `prep.yaml`: Cleans and preprocesses merged training data.
  - `train.yaml`: Trains emotion classification model.
  - `eval.yaml`: Evaluates trained model.
  - `compare_register.yaml`, `deploy_if_better.yaml`, etc.: Model comparison, conditional deployment.

- **environments/**  
  Environment specifications for reproducible pipeline execution.  
  - `emotion.yml`: Main Conda/Docker spec for all jobs (Python 3.9, PyTorch, Transformers, AzureML SDK, etc).

- **merge_feedback_component/**  
  Source and configuration for custom feedback merge logic.  
  Merges user feedback (e.g., “good corrections”) with latest training data for improved retraining.

- **pipeline_retrain.yml**  
  AzureML pipeline definition for continuous retraining, including all major steps:
  1. Merge feedback with train set
  2. Data preparation
  3. Model training
  4. Evaluation
  5. Model comparison & registration
  6. Conditional deployment (only if improved)
  All steps are fully modular and version-controlled.

---

## Justification for Scheduled Retraining

**Why schedule retraining (e.g., weekly) rather than on every feedback upload?**

### 1. Expected Feedback Rate

- **Early-stage/Academic Project:**  
  Fewer than 10–20 active users/week; feedback trickles in slowly.
- **User Behavior:**  
  Most feedback only submitted for visible misclassifications, not every prediction or every day.

### 2. Retraining Cost

- **Retraining on every new feedback upload:**  
  - **Pros:** Model adapts instantly.  
  - **Cons:** Expensive—Azure compute is used for every small feedback change, creating many redundant runs.
- **Scheduled retraining:**  
  - **Pros:** Batches feedback, reduces idle compute, predictable billing.  
  - **Cons:** Slight delay before corrections are reflected (acceptable for most real-world and academic use).

### 3. Best Practice: Feedback Batching

- **Industry Standard:**  
  Most content/NLP feedback systems retrain on a schedule, not on every update.
- **Resource Efficient:**  
  Less waste—each retrain uses a meaningful amount of new data, not just single changes.

### **Recommended Retraining Schedule**

- **Once per Week (e.g., Sunday Night):**
  - Batches all feedback for the week
  - 10+ new “good” corrections = more robust model update
  - Off-peak = lower Azure compute rates
  - If no feedback, skip run or run eval-only

- **If user base grows:**  
  Switch to 2x/week (>50 users), or daily/continuous retrain for enterprise production with large feedback volumes.

---

## Usage

1. Edit component YAMLs in `components/` to modify pipeline steps.
2. Edit `emotion.yml` to adjust the training/inference environment.
3. Update `pipeline_retrain.yml` for new data paths, model versions, or compute resources.
4. Run pipeline with Azure CLI:
   ```sh
   az ml job create --file pipeline_retrain.yml
