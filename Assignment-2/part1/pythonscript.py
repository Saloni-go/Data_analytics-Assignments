import duckdb

# Task 1

con = duckdb.connect("assignment2.db")
con.execute(open("task1.sql").read())

# Task 2

con.execute("INSERT INTO dim_brand SELECT * FROM read_csv_auto('M25_DA_A2_Part1/brands.csv')")
con.execute("INSERT INTO dim_category SELECT * FROM read_csv_auto('M25_DA_A2_Part1/categories.csv')")

con.execute("""
    INSERT INTO dim_customer
    SELECT customer_id, first_name, last_name, city, state, zip_code
    FROM read_csv_auto('M25_DA_A2_Part1/customers.csv')
""")

con.execute("""
    INSERT INTO dim_store
    SELECT store_id, store_name, phone, email, street, city, state, zip_code
    FROM read_csv_auto('M25_DA_A2_Part1/stores.csv')
""")

con.execute("""
    INSERT INTO dim_staff
    SELECT staff_id, first_name, last_name, email, phone, active, store_id
    FROM read_csv_auto('M25_DA_A2_Part1/staffs.csv')
""")

con.execute("""
    INSERT INTO dim_product
    SELECT product_id, product_name, brand_id, category_id, model_year, list_price
    FROM read_csv_auto('M25_DA_A2_Part1/products.csv')
""")

con.execute("""
    INSERT INTO dim_time
    SELECT DISTINCT
        order_date,
        EXTRACT(day FROM order_date) AS day,
        EXTRACT(month FROM order_date) AS month,
        EXTRACT(quarter FROM order_date) AS quarter,
        EXTRACT(year FROM order_date) AS year
    FROM read_csv_auto('M25_DA_A2_Part1/orders.csv', DATEFORMAT='%Y-%m-%d')
""")

con.execute("""
    INSERT INTO fact_order_items
    SELECT 
        oi.order_id,
        oi.item_id,
        oi.product_id,
        p.brand_id,
        p.category_id,
        o.customer_id,
        o.store_id,
        o.staff_id,
        o.order_date,
        oi.quantity,
        oi.list_price,
        oi.discount
    FROM read_csv_auto('M25_DA_A2_Part1/order_items.csv') AS oi
    JOIN read_csv_auto('M25_DA_A2_Part1/orders.csv', DATEFORMAT='%Y-%m-%d') AS o
        ON oi.order_id = o.order_id
    JOIN read_csv_auto('M25_DA_A2_Part1/products.csv') AS p
        ON oi.product_id = p.product_id
""")

print("All dimension and fact tables populated.")

# Task 3

queries = {
    "Q1_Total_Revenue_Year_Quarter_Month": """
        SELECT
            year,
            quarter,
            month,
            SUM(quantity * list_price * (1 - discount)) AS total_revenue
        FROM fact_order_items f
        JOIN dim_time t ON f.order_date = t.order_date
        GROUP BY ROLLUP (year, quarter, month)
        ORDER BY year, quarter, month;
    """,

    "Q2_Month_With_Highest_Sales_Per_Year": """
        WITH revenue_by_month AS (
            SELECT
                year,
                month,
                SUM(quantity * list_price * (1 - discount)) AS total_revenue,
                ROW_NUMBER() OVER (PARTITION BY year ORDER BY SUM(quantity * list_price * (1 - discount)) DESC) AS rn
            FROM fact_order_items f
            JOIN dim_time t ON f.order_date = t.order_date
            GROUP BY year, month
        )
        SELECT year, month, total_revenue
        FROM revenue_by_month
        WHERE rn = 1;
    """,

    "Q3_Revenue_By_Category": """
        SELECT
            c.category_name,
            SUM(f.quantity * f.list_price * (1 - f.discount)) AS total_revenue
        FROM fact_order_items f
        JOIN dim_product p ON f.product_id = p.product_id
        JOIN dim_category c ON p.category_id = c.category_id
        GROUP BY c.category_name
        ORDER BY total_revenue DESC;
    """,

    "Q4_Revenue_By_Category_And_Product": """
        SELECT
            c.category_name,
            p.product_name,
            SUM(f.quantity * f.list_price * (1 - f.discount)) AS total_revenue
        FROM fact_order_items f
        JOIN dim_product p ON f.product_id = p.product_id
        JOIN dim_category c ON p.category_id = c.category_id
        GROUP BY GROUPING SETS (
            (c.category_name),           
            (c.category_name, p.product_name)  
        )
        ORDER BY 
            c.category_name,
            total_revenue DESC;
    """,

    "Q5_Cube_Brand_Category_Year": """
        SELECT
            b.brand_name,
            c.category_name,
            t.year,
            SUM(f.quantity * f.list_price * (1 - f.discount)) AS total_revenue
        FROM fact_order_items f
        JOIN dim_product p ON f.product_id = p.product_id
        JOIN dim_brand b ON p.brand_id = b.brand_id
        JOIN dim_category c ON p.category_id = c.category_id
        JOIN dim_time t ON f.order_date = t.order_date
        GROUP BY CUBE (b.brand_name, c.category_name, t.year)
        ORDER BY b.brand_name, c.category_name, t.year;
    """,

    "Q6_Top_5_Customers": """
        SELECT
            cu.customer_id,
            cu.first_name || ' ' || cu.last_name AS customer_name,
            SUM(f.quantity * f.list_price * (1 - f.discount)) AS total_spent
        FROM fact_order_items f
        JOIN dim_customer cu ON f.customer_id = cu.customer_id
        GROUP BY cu.customer_id, customer_name
        ORDER BY total_spent DESC
        LIMIT 5;
    """,

    "Q7_Staff_Sales_By_Store": """
        SELECT
            s.store_id,
            s.first_name || ' ' || s.last_name AS staff_name,
            SUM(f.quantity * f.list_price * (1 - f.discount)) AS staff_sales
        FROM fact_order_items f
        JOIN dim_staff s ON f.staff_id = s.staff_id
        GROUP BY s.store_id, staff_name
        ORDER BY s.store_id, staff_sales DESC;
    """,

    "Q8_Cube_Category_Store_Year": """
        SELECT
            c.category_name,
            s.store_name,
            t.year,
            SUM(f.quantity * f.list_price * (1 - f.discount)) AS total_revenue
        FROM fact_order_items f
        JOIN dim_product p ON f.product_id = p.product_id
        JOIN dim_category c ON p.category_id = c.category_id
        JOIN dim_store s ON f.store_id = s.store_id
        JOIN dim_time t ON f.order_date = t.order_date
        GROUP BY CUBE (c.category_name, s.store_name, t.year)
        ORDER BY c.category_name, s.store_name, t.year;
    """
}

for name, sql in queries.items():
    print(f"Running: {name}")
    df = con.execute(sql).fetchdf()
    df.to_csv(f"{name}.csv", index=False)
    print(f"Saved: {name}.csv")

print("All queries executed and results saved.")
