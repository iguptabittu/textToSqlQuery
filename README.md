
# Fine-Tuning T5 for SQL Query Generation

This repository contains a Jupyter notebook for fine-tuning the T5 (Text-to-Text Transfer Transformer) model to generate SQL queries based on natural language questions and corresponding contexts (SQL `CREATE TABLE` statements).

### Dataset

The dataset consists of three columns:

- **Question**: A natural language question regarding a specific table or database schema.
- **Context**: A SQL `CREATE TABLE` statement representing the structure of a table.
- **Answer**: A valid SQL query answering the question based on the given context.

### Objective

The goal of fine-tuning the T5 model is to train it to generate SQL queries when provided with a question and context as input. The fine-tuned model will be able to generate valid SQL queries to retrieve information from a database.

---

### Steps to Fine-Tune T5

#### 1. Preprocessing the Data

- Combine the question and context to form a complete input prompt.
- Use the answer column as the target output (SQL query).
- Convert the preprocessed data into a format suitable for training using the Hugging Face `datasets` library.

#### 2. Fine-Tuning T5 Model

- Load the pre-trained `t5-small` model and tokenizer from the Hugging Face `transformers` library.
- Fine-tune the model using the dataset with the `Trainer` class.

#### 3. Training

- Set training parameters like batch size, learning rate, and number of epochs.
- Train the model on the dataset and evaluate its performance after each epoch.

#### 4. Inference

- After fine-tuning, the model can be tested with new questions and contexts to generate SQL queries.

---

### Requirements

- `transformers`: For working with pre-trained models and fine-tuning.
- `datasets`: For loading and preprocessing the dataset.
- `torch`: PyTorch for training the model.

To install the required libraries, run:

```bash
pip install transformers datasets torch
```

---

### Usage

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/fine-tune-t5-sql.git
```

2. **Run the Jupyter notebook** to fine-tune the T5 model:

```bash
jupyter notebook fine_tune_t5.ipynb
```

---

### Example: Inference

Once fine-tuned, the model can be used to generate SQL queries based on new questions and contexts. For example:

```python
test_question = "How many heads of the departments are older than 50?"
test_context = "CREATE TABLE head (age INTEGER)"
print(generate_answer(test_question, test_context))
```

Output:

```sql
SELECT COUNT(*) FROM head WHERE age > 50
```

---

### Fine-Tuning Process

- The model is fine-tuned using the `t5-small` architecture, but you can switch to `t5-base` or `t5-large` for improved performance.
- The training process includes tokenizing the input (question + context) and output (SQL query), and training the model over a specified number of epochs.
- After training, the model and tokenizer are saved for future use.

```python
model.save_pretrained('fine_tuned_t5')
tokenizer.save_pretrained('fine_tuned_t5')
```

---

### License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
```

