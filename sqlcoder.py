from llama_cpp import Llama

def main():
    # Hardcoded model path and parameters
    MODEL_PATH = "sqlcoder-7b-2.Q4_K_M.gguf"  # REPLACE THIS WITH YOUR ACTUAL MODEL PATH
    N_CTX = 2048
    N_THREADS = 6
    VERBOSE = True
    
    # Initialize the model with optimal settings for performance
    print(f"Loading model from {MODEL_PATH}...")
    model = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=0,      # CPU only
        n_ctx=N_CTX,         # Context size
        n_threads=N_THREADS, # CPU threads
        verbose=VERBOSE,
        n_batch=512,         # Batch size for more efficient processing
        use_mlock=True,      # Lock memory to prevent swapping
        use_mmap=True,       # Use memory mapping for faster loading
        logits_all=False     # Only compute logits for the last token
    )
    print("Model loaded successfully!")

    # Sample table schema
    table_schema = """
-- Customers table to store customer information
CREATE TABLE customers (
    customer_id INT PRIMARY KEY AUTO_INCREMENT,
    company_name VARCHAR(100) NOT NULL,
    contact_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    address VARCHAR(200),
    city VARCHAR(50),
    country VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Products table to store product information
CREATE TABLE products (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    product_name VARCHAR(100) NOT NULL,
    description TEXT,
    unit_price DECIMAL(10, 2) NOT NULL,
    stock_quantity INT NOT NULL DEFAULT 0,
    category VARCHAR(50),
    supplier_id INT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Orders table to store order headers
CREATE TABLE orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status ENUM('pending', 'processing', 'shipped', 'delivered', 'cancelled') DEFAULT 'pending',
    shipping_address VARCHAR(200),
    shipping_city VARCHAR(50),
    shipping_country VARCHAR(50),
    payment_method VARCHAR(50),
    total_amount DECIMAL(12, 2) NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Order details table to store items within each order
CREATE TABLE order_details (
    detail_id INT PRIMARY KEY AUTO_INCREMENT,
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity INT NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    discount DECIMAL(5, 2) DEFAULT 0,
    subtotal DECIMAL(12, 2) GENERATED ALWAYS AS (quantity * unit_price * (1 - discount)) STORED,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Inventory transactions to track stock movements
CREATE TABLE inventory_transactions (
    transaction_id INT PRIMARY KEY AUTO_INCREMENT,
    product_id INT NOT NULL,
    transaction_type ENUM('purchase', 'sale', 'adjustment', 'return') NOT NULL,
    quantity INT NOT NULL,
    reference_id INT COMMENT 'Order ID or other reference',
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
"""

    # SQLCoder prompt template
    prompt_template = """### Task
Generate a SQL query to answer the following question:
{question}

### Database Schema
The query will run on a database with the following schema:
{schema}

### SQL Query
```sql
"""

    while True:
        try:
            # Get user question
            user_question = input("\nEnter your question (or 'exit' to quit): ")
            if user_question.lower() in ['exit', 'quit']:
                break

            # Format the prompt
            prompt = prompt_template.format(
                question=user_question,
                schema=table_schema
            )

            # Generate SQL query with streaming for better UX
            print("\nGenerating SQL query...")
            
            print("\n--- Generated SQL Query ---")
            
            # Stream the response for better UX and automatic KV caching
            response = ""
            for chunk in model.create_completion(
                prompt,
                max_tokens=512,
                stop=["```"],
                echo=False,
                temperature=0.1,
                stream=True
            ):
                piece = chunk["choices"][0]["text"]
                response += piece
                print(piece, end="", flush=True)
            
            print("\n---------------------------")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
