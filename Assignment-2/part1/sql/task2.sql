-- .open assignment2.db
-- .read task2.sql

INSERT INTO dim_brand
SELECT * FROM read_csv_auto('M25_DA_A2_Part1/brands.csv');

INSERT INTO dim_category
SELECT * FROM read_csv_auto('M25_DA_A2_Part1/categories.csv');

INSERT INTO dim_customer
SELECT customer_id, first_name, last_name, city, state, zip_code
FROM read_csv_auto('M25_DA_A2_Part1/customers.csv');

INSERT INTO dim_store
SELECT store_id, store_name, phone, email, street, city, state, zip_code
FROM read_csv_auto('M25_DA_A2_Part1/stores.csv');

INSERT INTO dim_staff
SELECT staff_id, first_name, last_name, email, phone, active, store_id
FROM read_csv_auto('M25_DA_A2_Part1/staffs.csv');

INSERT INTO dim_product
SELECT product_id, product_name, brand_id, category_id, model_year, list_price
FROM read_csv_auto('M25_DA_A2_Part1/products.csv');

INSERT INTO dim_time
SELECT DISTINCT
    order_date,
    EXTRACT(day FROM order_date) AS day,
    EXTRACT(month FROM order_date) AS month,
    EXTRACT(quarter FROM order_date) AS quarter,
    EXTRACT(year FROM order_date) AS year
FROM read_csv_auto('M25_DA_A2_Part1/orders.csv', DATEFORMAT='%Y-%m-%d');

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
    ON oi.product_id = p.product_id;