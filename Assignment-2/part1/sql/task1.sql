-- DIMENSION TABLES

CREATE TABLE dim_customer (
    customer_id INTEGER PRIMARY KEY,
    first_name VARCHAR,
    last_name VARCHAR,
    city VARCHAR,
    state VARCHAR,
    zip_code VARCHAR
);

CREATE TABLE dim_store (
    store_id INTEGER PRIMARY KEY,
    store_name VARCHAR,
    phone VARCHAR,
    email VARCHAR,
    street VARCHAR,
    city VARCHAR,
    state VARCHAR,
    zip_code VARCHAR
);

CREATE TABLE dim_staff (
    staff_id INTEGER PRIMARY KEY,
    first_name VARCHAR,
    last_name VARCHAR,
    email VARCHAR,
    phone VARCHAR,
    active BOOLEAN,
    store_id INTEGER
);

CREATE TABLE dim_brand (
    brand_id INTEGER PRIMARY KEY,
    brand_name VARCHAR
);

CREATE TABLE dim_category (
    category_id INTEGER PRIMARY KEY,
    category_name VARCHAR
);

CREATE TABLE dim_product (
    product_id INTEGER PRIMARY KEY,
    product_name VARCHAR,
    brand_id INTEGER,
    category_id INTEGER,
    model_year INTEGER,
    list_price DECIMAL(10,2)
);

CREATE TABLE dim_time (
    order_date DATE PRIMARY KEY,
    day INTEGER,
    month INTEGER,
    quarter INTEGER,
    year INTEGER
);

-- FACT TABLE

CREATE TABLE fact_order_items (
    order_id INTEGER,
    item_id INTEGER,
    product_id INTEGER,
    brand_id INTEGER,
    category_id INTEGER,
    customer_id INTEGER,
    store_id INTEGER,
    staff_id INTEGER,
    order_date DATE,
    quantity INTEGER,
    list_price DECIMAL(10,2),
    discount DECIMAL(4,2),
    PRIMARY KEY(order_id, item_id)
);